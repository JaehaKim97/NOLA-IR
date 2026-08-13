from typing import List

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from pipeline.segmentation.base import BasePipeline
from utils.common import calculate_psnr_pt, instantiate_from_config, load_network, wavelet_reconstruction, calculate_psnr_list
from utils.segmentation import calculate_mat, compute_iou, mask2rgb


class EDTRPipeline(BasePipeline):

    def init_models(self):
        self.init_timesteps()
        self.init_scheduler()
        self.init_res_model(train=False)
        self.init_text_models()
        self.init_vae()
        self.init_cldm()
        self.init_seg_model()

    def init_optimizers(self):
        if not self.args.eval_only:
            opt_config = self.cfg.train.optimizer.edtr
            if opt_config.type.lower() == "adamw":
                optimizer = torch.optim.AdamW
            else:
                raise NotImplementedError(f"{opt_config.type} is Not supported optimizer for EDTR")

            self.gen_params = list(filter(lambda p: p.requires_grad, self.cldm.controlnet.parameters()))
            if self.train_vae:
                self.gen_params += list(filter(lambda p: p.requires_grad, self.vae.parameters()))
            self.opt_gen = optimizer(self.gen_params, **opt_config.kwargs)
            self.sch_gen = CosineAnnealingLR(self.opt_gen, T_max=self.cfg.train.train_steps, eta_min=1e-7)

            opt_config = self.cfg.train.optimizer.seg_model
            if opt_config.type.lower() == "sgd":
                optimizer = torch.optim.SGD
            elif opt_config.type.lower() == "adamw":
                optimizer = torch.optim.AdamW
            else:
                raise NotImplementedError(f"{opt_config.type} is Not supported optimizer for seg_model")

            self.seg_model_params = list(filter(lambda p: p.requires_grad, self.seg_model.parameters()))
            self.opt_seg_model = optimizer(self.seg_model_params, **opt_config.kwargs)
            self.sch_seg_model = CosineAnnealingLR(self.opt_seg_model, T_max=self.cfg.train.train_steps, eta_min=1e-7)

    def prepare_all(self):
        attrs = ["res_model", "vae", "text_encoder", "cldm", "seg_model", "seg_hq_model", "val_dataloader"]
        if not self.args.eval_only:
            attrs += ["opt_gen", "sch_gen", "opt_seg_model", "sch_seg_model", "train_dataloader"]

        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)

    def partial_diff_and_sample(self, x, c_img, c_txt, t=None, mode="eval"):
        noise = torch.randn_like(x)

        for i, step in enumerate(sorted(self.timesteps, reverse=True)):
            if mode == "eval":
                t = torch.full((x.size(0),), step, device=self.device, dtype=torch.long)
            if i == 0:
                x = self.scheduler.add_noise(x, noise, t)
            
            eps = self.cldm(x, t, {"c_img": c_img, "c_txt": c_txt})

            index = self.t2index(t)
            x0_hat = self.spaced_sampler._predict_xstart_from_eps(x, index, eps)
            if mode == "train":
                x = x0_hat
                break
            model_mean, model_var, _ = self.spaced_sampler.q_posterior_mean_variance(x0_hat, x, index)
            nonzero_mask = (index != 0).float().view(-1, *([1] * (x.ndim - 1)))
            x = model_mean + nonzero_mask * torch.sqrt(model_var) * torch.randn_like(x)
        return x

    def calculate_hlf_loss(self, feat_res, feat_gt, feat_hq_res, feat_hq_gt, target_layers=None, margin=0.0):
        if target_layers is None:
            target_layers = self.get_target_layers()

        loss_hlf = sum(
            self.masked_l1(feat_res[k], feat_gt[k], margin)
            + self.masked_l1(feat_hq_res[k], feat_hq_gt[k], margin)
            for k in target_layers
        ) / (2 * len(target_layers))
        return loss_hlf

    def train(self):
        self.loss_records = dict(HLF=[], SEG=[], FM=[])
        self.on_training_start()

        while self.global_step < self.max_steps:
            pbar = self.make_pbar(total=len(self.train_dataloader))
            for batch in self.train_dataloader:
                self.prepare_batch_inputs(batch, transform=True)
                b = self.gt.size(0)
                bh = b // 2

                with self.accelerator.autocast():
                    if hasattr(self, "cldm"):
                        self.cldm.train()
                    if self.train_vae:
                        self.vae.train()
                    self.seg_model.eval().requires_grad_(False)

                    pre_res, z_pre_res, c_txt = self.prepare_condition()
                    t = self.timesteps_pt[torch.randint(0, len(self.timesteps), (b,), device=self.device)]
                    z_res = self.partial_diff_and_sample(z_pre_res, z_pre_res, c_txt, t, mode="train")
                    res = wavelet_reconstruction(self.decode_image(z_res), pre_res)

                    _, feat_gt = self.seg_model(self.gt, return_feat=True)
                    _, feat_res = self.seg_model(res, return_feat=True)
                    with torch.no_grad():
                        _, feat_hq_gt = self.seg_hq_model(self.gt, return_feat=True)
                    _, feat_hq_res = self.seg_hq_model(res, return_feat=True)
                    loss_hlf = self.calculate_hlf_loss(feat_res, feat_gt, feat_hq_res, feat_hq_gt)
                    loss_hlf *= self.cfg.train.weight.hlf

                self.opt_gen.zero_grad()
                self.accelerator.backward(loss_hlf)
                self.opt_gen.step()
                self.sch_gen.step()

                with self.accelerator.autocast():
                    if hasattr(self, "cldm"):
                        self.cldm.eval()
                    if self.train_vae:
                        self.vae.eval()
                    self.seg_model.train().requires_grad_(True)

                    with torch.no_grad():
                        z_res = self.partial_diff_and_sample(z_pre_res[:bh], z_pre_res[:bh], c_txt[:bh])
                        res = wavelet_reconstruction(self.decode_image(z_res), pre_res[:bh])

                    pred, feat_mix = self.seg_model(torch.cat((res, self.gt[bh:]), dim=0), return_feat=True)
                    loss_seg = F.cross_entropy(pred["out"], self.mask, ignore_index=255) * self.cfg.train.weight.seg
                    loss_fm = F.l1_loss(feat_mix["C5"], feat_hq_gt["C5"]) * self.cfg.train.weight.fm

                self.opt_seg_model.zero_grad()
                self.accelerator.backward(loss_seg + loss_fm)
                self.opt_seg_model.step()
                self.sch_seg_model.step()

                self.global_step += 1
                self.loss_records["HLF"].append(loss_hlf.item())
                self.loss_records["SEG"].append(loss_seg.item())
                self.loss_records["FM"].append(loss_fm.item())
                pbar.update(1)
                pbar.set_description(
                    f"Epoch: {self.epoch:04d}, Steps: {self.global_step:07d}, "
                    f"HLF: {loss_hlf.item():.3f}, SEG: {loss_seg.item():.3f}, FM: {loss_fm.item():.3f}"
                )

                if self.global_step % self.cfg.train.log_every == 0 or (self.args.debug):
                    self.log_training_metrics()
                if self.global_step % self.cfg.train.image_every == 0 or (self.args.debug):
                    self.seg_model.eval()
                    with self.accelerator.autocast(), torch.no_grad():
                        pred = self.seg_model(res)
                    self.seg_model.train()
                    img_dict={"pre_restored": pre_res, "restored": res, "pred": mask2rgb(pred["out"].argmax(1))}
                    self.log_images(img_dict, N=2)
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
        if hasattr(self, "cldm"):
            self.cldm.eval()
        if self.train_vae:
            self.vae.eval()
        self.seg_model.eval()

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
                z_res = self.partial_diff_and_sample(z_pre_res, z_pre_res ,c_txt)
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
            if hasattr(self, "cldm"):
                self.cldm.train()
            if self.train_vae:
                self.vae.train()
            self.seg_model.train()
