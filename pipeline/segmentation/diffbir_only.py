from typing import List

import torch
from diffusers import DDPMScheduler
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from pipeline.segmentation.base import BasePipeline
from utils.common import calculate_psnr_list, wavelet_reconstruction
from utils.segmentation import calculate_mat, compute_iou
from utils.sampler import SpacedSampler


class DiffBIROnlyPipeline(BasePipeline):

    def init_models(self):
        self.init_scheduler_and_timesteps()
        self.init_res_model(train=False)
        self.init_text_models()
        self.init_vae()
        self.init_cldm()
        self.init_seg_model(train=False)

    def init_scheduler_and_timesteps(self):
        self.t_start = self.cfg.model.diffusion.start
        self.num_timesteps = self.cfg.model.diffusion.num_timesteps
        self.scheduler = DDPMScheduler.from_pretrained(self.cfg.model.sd_path, subfolder="scheduler")
        self.spaced_sampler = SpacedSampler(self.scheduler.betas)
        self.spaced_sampler.make_schedule(self.num_timesteps)
        self.timesteps = sorted(list(self.spaced_sampler.timesteps))
        self.spaced_sampler.to(self.device)

    def init_optimizers(self):
        if not self.args.eval_only:
            opt_config = self.cfg.train.optimizer.diffbir
            if opt_config.type.lower() == "adamw":
                optimizer = torch.optim.AdamW
            else:
                raise NotImplementedError(f"{opt_config.type} is Not supported optimizer for DiffBIR")

            self.gen_params = list(filter(lambda p: p.requires_grad, self.cldm.controlnet.parameters()))
            self.opt_gen = optimizer(self.gen_params, **opt_config.kwargs)
            self.sch_gen = CosineAnnealingLR(self.opt_gen, T_max=self.cfg.train.train_steps, eta_min=1e-7)

    def prepare_all(self):
        attrs = ["res_model", "vae", "text_encoder", "cldm", "seg_model", "seg_hq_model", "val_dataloader"]
        if not self.args.eval_only:
            attrs += ["opt_gen", "sch_gen", "train_dataloader"]

        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)

    def diff_and_sample(self, x, c_img, c_txt, t=None, noise=None, mode="eval"):
        if mode == "train":
            x = self.scheduler.add_noise(x, noise, t)

        for i, step in enumerate(sorted(self.timesteps, reverse=True)):
            if mode == "eval":
                t = torch.full((x.size(0),), step, device=self.device, dtype=torch.long)

            eps = self.cldm(x, t, {"c_img": c_img, "c_txt": c_txt})

            if mode == "train":
                x = eps
                break
            index = self.t2index(t)
            x0_hat = self.spaced_sampler._predict_xstart_from_eps(x, index, eps)
            model_mean, model_var, _ = self.spaced_sampler.q_posterior_mean_variance(x0_hat, x, index)
            nonzero_mask = (index != 0).float().view(-1, *([1] * (x.ndim - 1)))
            x = model_mean + nonzero_mask * torch.sqrt(model_var) * torch.randn_like(x)
        return x

    def train(self):
        self.loss_records = dict(L2=[])
        self.on_training_start()

        while self.global_step < self.max_steps:
            pbar = self.make_pbar(total=len(self.train_dataloader))
            for batch in self.train_dataloader:
                self.prepare_batch_inputs(batch, transform=True)
                b = self.gt.size(0)

                with self.accelerator.autocast():
                    pre_res, z_pre_res, c_txt, z_gt = self.prepare_condition(return_gt=True)
                    t = torch.randint(0, self.t_start + 1, (b,), device=self.device)
                    noise = torch.randn_like(z_pre_res)
                    pred_noise = self.diff_and_sample(z_gt, z_pre_res, c_txt, t, noise, mode="train")
                    loss_l2 = F.mse_loss(pred_noise, noise)

                self.opt_gen.zero_grad()
                self.accelerator.backward(loss_l2)
                self.opt_gen.step()
                self.sch_gen.step()

                self.global_step += 1
                self.loss_records["L2"].append(loss_l2.item())
                pbar.update(1)
                pbar.set_description(
                    f"Epoch: {self.epoch:04d}, Steps: {self.global_step:07d}, L2: {loss_l2.item():.3f}"
                )

                if self.global_step % self.cfg.train.log_every == 0 or (self.args.debug):
                    self.log_training_metrics()
                if self.global_step % self.cfg.train.image_every == 0 or (self.args.debug):
                    self.cldm.eval()
                    with self.accelerator.autocast(), torch.no_grad():
                        z_res = self.diff_and_sample(torch.randn_like(z_pre_res), z_pre_res, c_txt)
                        res = wavelet_reconstruction(self.decode_image(z_res), pre_res)
                    self.log_images({"pre_restored": pre_res, "restored": res}, N=4)
                    self.cldm.train()
                if self.global_step % self.cfg.train.ckpt_every == 0 or (self.args.debug):
                    self.save_checkpoints()
                if self.global_step % self.cfg.val.val_every == 0 or (self.args.debug):
                    self.evaluate()

                if self.global_step >= self.max_steps:
                    break

            pbar.close()
            self.epoch += 1

        self.save_checkpoints(last=True)
        self.accelerator.end_training()

    @torch.no_grad()
    def evaluate(self):
        self.cldm.eval()

        pbar = self.make_pbar(total=len(self.val_dataloader), desc="Validation", leave=False)
        confmat = torch.zeros((self.n_classes, self.n_classes), device=self.device)
        tfd_list: List[torch.Tensor] = []
        psnr_list: List[torch.Tensor] = []

        for batch in self.val_dataloader:
            self.prepare_batch_inputs(batch)
            assert self.gt.size(0) == 1

            with self.accelerator.autocast():
                lq_padded = self.pad_image(self.lq, multiple=64)
                pre_res, z_pre_res, c_txt = self.prepare_condition(lq=lq_padded)
                z_res = self.diff_and_sample(torch.randn_like(z_pre_res), z_pre_res ,c_txt)
                res_padded = wavelet_reconstruction(self.decode_image(z_res), pre_res)
                res = res_padded[..., :self.lq.size(2), :self.lq.size(3)]
                pred = self.seg_model(res)

                feat_hq_gt = self.seg_hq_model(self.gt, return_feat=True)[1]
                feat_hq_res = self.seg_hq_model(res, return_feat=True)[1]

            mat = calculate_mat(self.mask.flatten(), pred['out'].argmax(1).flatten(), n=self.n_classes).unsqueeze(0)
            tfd = self.calculate_tfd(feat_hq_res, feat_hq_gt)
            psnr = calculate_psnr_pt(res, self.gt, crop_border=0)

            mat, tfd, psnr = self.accelerator.gather_for_metrics((mat, tfd, psnr))
            if self.accelerator.is_local_main_process:
                confmat += mat.sum(0)
                tfd_list += tfd.tolist()
                psnr_list += psnr.tolist()

            if self.args.save_img:
                self.save_masked_image(res, pred["out"].argmax(1), self.filename)

            pbar.update(1)
        pbar.close()

        if self.accelerator.is_local_main_process:
            miou = compute_iou(confmat).mean().item() * 100
            avg_tfd = torch.tensor(tfd_list).mean().item()
            avg_psnr = torch.tensor(psnr_list).mean().item()
            for tag, val in [("val/miou", miou), ("val/tfd", avg_tfd), ("val/psnr", avg_psnr),]:
                self.log(f"{tag}: {val:.4f}")
                if not self.args.eval_only:
                    self.writer.add_scalar(tag, val, self.global_step)

        self.accelerator.wait_for_everyone()
        if self.args.eval_only:
            self.accelerator.end_training()
        else:
            torch.cuda.empty_cache()
            self.cldm.train()
