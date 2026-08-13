from typing import List

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from pipeline.optical_character_recognition.base import BasePipeline
from utils.common import calculate_psnr_pt, instantiate_from_config, load_network, log_txt_as_img, wavelet_reconstruction
from utils.optical_character_recognition import calculate_char_accuracy


class EDTRPipeline(BasePipeline):

    def init_models(self):
        self.init_timesteps()
        self.init_scheduler()
        self.init_converter()
        self.init_res_model(train=False)
        self.init_text_models()
        self.init_vae()
        self.init_cldm()
        self.init_ocr_model()
        self.init_ocr_hq_model()

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

            opt_config = self.cfg.train.optimizer.ocr_model
            if opt_config.type.lower() == "sgd":
                optimizer = torch.optim.SGD
            elif opt_config.type.lower() == "adam":
                optimizer = torch.optim.Adam
            else:
                raise NotImplementedError(f"{opt_config.type} is Not supported optimizer for ocr_model")

            self.ocr_model_params = list(filter(lambda p: p.requires_grad, self.ocr_model.parameters()))
            self.opt_ocr_model = optimizer(self.ocr_model_params, **opt_config.kwargs)
            self.sch_ocr_model = CosineAnnealingLR(self.opt_ocr_model, T_max=self.cfg.train.train_steps, eta_min=1e-7)

    def prepare_all(self):
        attrs = ["res_model", "vae", "text_encoder", "cldm", "ocr_hq_model", "ocr_model", "val_dataloader"]
        if not self.args.eval_only:
            attrs += ["opt_gen", "sch_gen", "opt_ocr_model", "sch_ocr_model", "train_dataloader"]

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
        self.loss_records = dict(HLF=[], OCR=[], FM=[])
        self.on_training_start()

        while self.global_step < self.max_steps:
            pbar = self.make_pbar(total=len(self.train_dataloader))
            for batch in self.train_dataloader:
                self.prepare_batch_inputs(batch, transform=True)
                assert self.gt.size(0) % (8 * 2) == 0, "Batch size should be divisible by 8(row) * 2(col) for tiling"
                bh = self.gt.size(0) // 2
                b_tile = self.gt.size(0) // (8 * 2)
                bh_tile = int(np.ceil(b_tile / 2))

                with self.accelerator.autocast():
                    if hasattr(self, "cldm"):
                        self.cldm.train()
                    if self.train_vae:
                        self.vae.train()
                    self.ocr_model.eval().requires_grad_(False)

                    lq_tile = self.img2tile(self.lq)
                    pre_res_tile, z_pre_res_tile, c_txt = self.prepare_condition(lq=lq_tile, batch_size=b_tile)
                    t = self.timesteps_pt[torch.randint(0, len(self.timesteps), (b_tile,), device=self.device)]
                    z_res_tile = self.partial_diff_and_sample(z_pre_res_tile, z_pre_res_tile, c_txt, t, mode="train")
                    res_tile = wavelet_reconstruction(self.decode_image(z_res_tile), pre_res_tile)
                    res = self.tile2img(res_tile)

                    _, feat_gt = self.ocr_model(self.gt, is_train=False, return_feat=True)
                    _, feat_res = self.ocr_model(res, is_train=False, return_feat=True)
                    with torch.no_grad():
                        _, feat_hq_gt = self.ocr_hq_model(self.gt, is_train=False, return_feat=True)
                    _, feat_hq_res = self.ocr_hq_model(res, is_train=False, return_feat=True)
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
                    self.ocr_model.train().requires_grad_(True)

                    with torch.no_grad():
                        z_res_tile = self.partial_diff_and_sample(z_pre_res_tile[:bh_tile], z_pre_res_tile[:bh_tile], c_txt[:bh_tile])
                        res_tile = wavelet_reconstruction(self.decode_image(z_res_tile), pre_res_tile[:bh_tile])
                        res = self.tile2img(res_tile)

                    pred, feat_mix = self.ocr_model(torch.cat((res[:bh], self.gt[bh:]), dim=0), self.text, return_feat=True)
                    loss_ocr = self.calculate_ocr_loss(pred) * self.cfg.train.weight.ocr
                    loss_fm = F.l1_loss(feat_mix["L14"], feat_hq_gt["L14"]) * self.cfg.train.weight.fm

                self.opt_ocr_model.zero_grad()
                self.accelerator.backward(loss_ocr + loss_fm)
                self.opt_ocr_model.step()
                self.sch_ocr_model.step()

                self.global_step += 1
                self.loss_records["HLF"].append(loss_hlf.item())
                self.loss_records["OCR"].append(loss_ocr.item())
                self.loss_records["FM"].append(loss_fm.item())
                pbar.update(1)
                pbar.set_description(
                    f"Epoch: {self.epoch:04d}, Steps: {self.global_step:07d}, "
                    f"HLF: {loss_hlf.item():.3f}, OCR: {loss_ocr.item():.3f}, FM: {loss_fm.item():.3f}"
                )

                if self.global_step % self.cfg.train.log_every == 0 or (self.args.debug):
                    self.log_training_metrics()
                if self.global_step % self.cfg.train.image_every == 0 or (self.args.debug):
                    self.ocr_model.eval()
                    with self.accelerator.autocast(), torch.no_grad():
                        pred = self.ocr_model(res, is_train=False)
                        pred_label = self.convert2label(pred, batch_size=bh)
                    img_dict = {
                        "pre_restored": self.tile2img(pre_res_tile),
                        "restored": res,
                        "pred_label": log_txt_as_img((128, 32), pred_label)
                    }
                    self.log_images(img_dict, N=48)
                    self.ocr_model.train()
                if self.global_step % self.cfg.train.ckpt_every == 0 or (self.args.debug):
                    self.save_checkpoints()
                if self.global_step % self.cfg.val.val_every == 0 or (self.global_step == 5000) or (self.args.debug):
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
        self.ocr_model.eval()

        pbar = self.make_pbar(total=len(self.val_dataloader), desc="Validation", leave=False)
        word_correct_list: List[torch.Tensor] = []
        char_correct_list: List[torch.Tensor] = []
        psnr_list: List[torch.Tensor] = []

        for batch in self.val_dataloader:
            self.prepare_batch_inputs(batch)
            B = self.lq.size(0)
            if B % 16 != 0:
                # for the last batch in the dataloader, we need to pad it to make it divisible by 16 for tiling
                self.lq = torch.cat([self.lq, self.lq[-1:].expand(((-B % 16), *self.lq.shape[1:]))], dim=0)

            with self.accelerator.autocast():
                lq_tile = self.img2tile(self.lq)
                pre_res_tile, z_pre_res_tile, c_txt = self.prepare_condition(lq=lq_tile, batch_size=lq_tile.size(0))
                z_res_tile = self.partial_diff_and_sample(z_pre_res_tile, z_pre_res_tile, c_txt)
                res_tile = wavelet_reconstruction(self.decode_image(z_res_tile), pre_res_tile)
                res = self.tile2img(res_tile)[:B]
                pred = self.ocr_model(res, is_train=False)
                pred_label = self.convert2label(pred, batch_size=B)

            word_correct = torch.Tensor((np.array(pred_label) == np.array(self.label))).to(self.device)
            char_correct = torch.tensor(calculate_char_accuracy(pred_label, self.label)).to(self.device)
            
            gt, res, word_correct, char_correct = self.accelerator.gather_for_metrics(
                (self.gt, res, word_correct, char_correct)
            )
            if self.accelerator.is_local_main_process:
                word_correct_list += word_correct.tolist()
                char_correct_list += char_correct.tolist()
                psnr_list += calculate_psnr_pt(res, gt, crop_border=0).detach().cpu().float().tolist()

            if self.args.save_img:
                self.save_batch_images(res, self.filename, self.label, pred_label, flatten=True)

            pbar.update(1)
        pbar.close()

        if self.accelerator.is_local_main_process:
            avg_word_acc = torch.tensor(word_correct_list).mean().item()
            avg_char_acc = torch.tensor(char_correct_list).mean().item()
            avg_psnr = torch.tensor(psnr_list).mean().item()
            for tag, val in [("val/word_acc", avg_word_acc), ("val/char_acc", avg_char_acc), ("val/psnr", avg_psnr)]:
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
            self.ocr_model.train()
