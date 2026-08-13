from typing import List

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from pipeline.classification.base import BasePipeline
from utils.classification import calculate_accuracy
from utils.common import calculate_psnr_pt, instantiate_from_config, load_network, wavelet_reconstruction


class EDTRPipeline(BasePipeline):

    def init_models(self):
        self.init_timesteps()
        self.init_scheduler()
        self.init_res_model(train=False)
        self.init_text_models()
        self.init_vae()
        self.init_cldm()
        self.init_cls_model()

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

            opt_config = self.cfg.train.optimizer.cls_model
            if opt_config.type.lower() == "sgd":
                optimizer = torch.optim.SGD
            elif opt_config.type.lower() == "adamw":
                optimizer = torch.optim.AdamW
            else:
                raise NotImplementedError(f"{opt_config.type} is Not supported optimizer for cls_model")

            self.cls_model_params = list(filter(lambda p: p.requires_grad, self.cls_model.parameters()))
            self.opt_cls_model = optimizer(self.cls_model_params, **opt_config.kwargs)
            self.sch_cls_model = CosineAnnealingLR(self.opt_cls_model, T_max=self.cfg.train.train_steps, eta_min=1e-7)

    def prepare_all(self):
        attrs = ["res_model", "vae", "text_encoder", "cldm", "cls_model", "cls_hq_model", "val_dataloader"]
        if not self.args.eval_only:
            attrs += ["opt_gen", "sch_gen", "opt_cls_model", "sch_cls_model", "train_dataloader"]

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
        self.loss_records = dict(HLF=[], CLS=[], FM=[])
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
                    self.cls_model.eval().requires_grad_(False)

                    pre_res, z_pre_res, c_txt = self.prepare_condition(batch_size=bh)
                    t = self.timesteps_pt[torch.randint(0, len(self.timesteps), (bh,), device=self.device)]
                    z_res = self.partial_diff_and_sample(z_pre_res, z_pre_res, c_txt, t, mode="train")
                    res = wavelet_reconstruction(self.decode_image(z_res), pre_res)

                    _, feat_gt = self.cls_model(self.gt[:bh], return_feat=True)
                    _, feat_res = self.cls_model(res, return_feat=True)
                    with torch.no_grad():
                        _, feat_hq_gt = self.cls_hq_model(self.gt[:bh], return_feat=True)
                    _, feat_hq_res = self.cls_hq_model(res, return_feat=True)
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
                    self.cls_model.train().requires_grad_(True)

                    with torch.no_grad():
                        z_res = self.partial_diff_and_sample(z_pre_res, z_pre_res, c_txt)
                        res = wavelet_reconstruction(self.decode_image(z_res), pre_res)

                    pred, feat_mix = self.cls_model(torch.cat((res, self.gt[bh:]), dim=0), return_feat=True)
                    loss_cls = F.cross_entropy(pred, self.label) * self.cfg.train.weight.cls
                    with torch.no_grad():
                        _, feat_hq_gt = self.cls_hq_model(self.gt, return_feat=True)
                    loss_fm = F.l1_loss(feat_mix["L4"], feat_hq_gt["L4"]) * self.cfg.train.weight.fm

                self.opt_cls_model.zero_grad()
                self.accelerator.backward(loss_cls + loss_fm)
                self.opt_cls_model.step()
                self.sch_cls_model.step()

                self.global_step += 1
                self.loss_records["HLF"].append(loss_hlf.item())
                self.loss_records["CLS"].append(loss_cls.item())
                self.loss_records["FM"].append(loss_fm.item())
                pbar.update(1)
                pbar.set_description(
                    f"Epoch: {self.epoch:04d}, Steps: {self.global_step:07d}, "
                    f"HLF: {loss_hlf.item():.3f}, CLS: {loss_cls.item():.3f}, FM: {loss_fm.item():.3f}"
                )

                if self.global_step % self.cfg.train.log_every == 0 or (self.args.debug):
                    self.log_training_metrics()
                if self.global_step % self.cfg.train.image_every == 0 or (self.args.debug):
                    self.log_images(img_dict={"pre_restored": pre_res, "restored": res}, N=4)
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
        self.cls_model.eval()

        pbar = self.make_pbar(total=len(self.val_dataloader), desc="Validation", leave=False)
        acc1_list: List[torch.Tensor] = []
        tfd_list: List[torch.Tensor] = []
        psnr_list: List[torch.Tensor] = []

        for batch in self.val_dataloader:
            self.prepare_batch_inputs(batch)

            with self.accelerator.autocast():
                pre_res, z_pre_res, c_txt = self.prepare_condition()
                z_res = self.partial_diff_and_sample(z_pre_res, z_pre_res, c_txt)
                res = wavelet_reconstruction(self.decode_image(z_res), pre_res)
                pred = self.cls_model(res)

                feat_hq_res = self.cls_hq_model(res, return_feat=True)[1]
                feat_hq_gt = self.cls_hq_model(self.gt, return_feat=True)[1]

            tfd = self.calculate_tfd(feat_hq_res, feat_hq_gt)
            psnr = calculate_psnr_pt(res, self.gt, crop_border=0)

            pred, label, tfd, psnr = self.accelerator.gather_for_metrics((pred, self.label, tfd, psnr))
            if self.accelerator.is_local_main_process:
                acc1_list += [calculate_accuracy(pred, label, topk=(1, 5))[0]] * pred.size(0)
                tfd_list += tfd.tolist()
                psnr_list += psnr.tolist()

            if self.args.save_img:
                self.save_batch_images(res, self.filename, self.label, pred, flatten=True)

            pbar.update(1)
        pbar.close()

        if self.accelerator.is_local_main_process:
            avg_acc1 = torch.tensor(acc1_list).mean().item()
            avg_tfd = torch.tensor(tfd_list).mean().item()
            avg_psnr = torch.tensor(psnr_list).mean().item()
            for tag, val in [("val/acc1", avg_acc1), ("val/tfd", avg_tfd), ("val/psnr", avg_psnr)]:
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
            self.cls_model.train()
