from typing import List

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from pipeline.segmentation.diffbir_only import DiffBIROnlyPipeline
from utils.common import calculate_psnr_pt, wavelet_reconstruction
from utils.segmentation import calculate_mat, compute_iou, mask2rgb


class DiffBIRFixPipeline(DiffBIROnlyPipeline):

    def init_models(self):
        self.init_scheduler_and_timesteps()
        self.init_res_model(train=False)
        self.init_text_models()
        self.init_vae()
        self.init_cldm(train=False)
        self.init_seg_model()

    def init_optimizers(self):
        if not self.args.eval_only:
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
            attrs += ["opt_seg_model", "sch_seg_model", "train_dataloader"]

        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)

    def train(self):
        self.loss_records = dict(SEG=[])
        self.on_training_start()

        while self.global_step < self.max_steps:
            pbar = self.make_pbar(total=len(self.train_dataloader))
            for batch in self.train_dataloader:
                self.prepare_batch_inputs(batch, transform=True)
                b = self.gt.size(0)
                if self.cfg.train.use_half_batch:
                    b = b // 2

                with self.accelerator.autocast():
                    with torch.no_grad():
                        pre_res, z_pre_res, c_txt = self.prepare_condition(batch_size=b)
                        z_res = self.diff_and_sample(torch.randn_like(z_pre_res), z_pre_res, c_txt)
                        res = wavelet_reconstruction(self.decode_image(z_res), pre_res)

                    pred = self.seg_model(torch.cat((res, self.gt[b:]), dim=0))
                    loss_seg = F.cross_entropy(pred["out"], self.mask, ignore_index=255)

                self.opt_seg_model.zero_grad()
                self.accelerator.backward(loss_seg)
                self.opt_seg_model.step()
                self.sch_seg_model.step()

                self.global_step += 1
                self.loss_records["SEG"].append(loss_seg.item())
                pbar.update(1)
                pbar.set_description(
                    f"Epoch: {self.epoch:04d}, Steps: {self.global_step:07d}, SEG: {loss_seg.item():.3f}"
                )

                if self.global_step % self.cfg.train.log_every == 0 or (self.args.debug):
                    self.log_training_metrics()
                if self.global_step % self.cfg.train.image_every == 0 or (self.args.debug):
                    self.seg_model.eval()
                    with self.accelerator.autocast(), torch.no_grad():
                        pred = self.seg_model(res)
                    img_dict={"pre_restored": pre_res, "restored": res, "pred": mask2rgb(pred["out"].argmax(1))}
                    self.log_images(img_dict, N=8)
                    self.seg_model.train()
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
            self.seg_model.train()
