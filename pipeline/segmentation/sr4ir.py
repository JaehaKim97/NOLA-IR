from typing import List

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from pipeline.segmentation.base import BasePipeline
from utils.common import calculate_psnr_pt
from utils.segmentation import calculate_mat, compute_iou, mask2rgb


class SR4IRPipeline(BasePipeline):

    def init_models(self):
        self.init_res_model()
        self.init_seg_model()

    def init_optimizers(self):
        if not self.args.eval_only:
            opt_config = self.cfg.train.optimizer.res_model
            if opt_config.type.lower() == "adamw":
                optimizer = torch.optim.AdamW
            else:
                raise NotImplementedError(f"{opt_config.type} is Not supported optimizer for res_model")

            self.res_model_params = list(filter(lambda p: p.requires_grad, self.res_model.parameters()))
            self.opt_res_model = optimizer(self.res_model_params, **opt_config.kwargs)
            self.sch_res_model = CosineAnnealingLR(self.opt_res_model, T_max=self.cfg.train.train_steps, eta_min=1e-7)

            opt_config = self.cfg.train.optimizer.seg_model
            if opt_config.type.lower() == "sgd":
                optimizer = torch.optim.SGD
            elif opt_config.type.lower() == "adamw":
                optimizer = torch.optim.AdamW
            else:
                raise NotImplementedError(f"{opt_config.type} is Not supported optimizer for seg_model")

            self.seg_model_params = list(filter(lambda p: p.requires_grad, self.seg_model.parameters()))
            self.opt_seg_model = optimizer(self.seg_model_params, **opt_config.kwargs,)
            self.sch_seg_model = CosineAnnealingLR(self.opt_seg_model, T_max=self.cfg.train.train_steps, eta_min=1e-7)

    def prepare_all(self):
        attrs = ["res_model", "seg_model", "seg_hq_model", "val_dataloader"]
        if not getattr(self.args, "eval_only", False):
            attrs += ["opt_res_model", "sch_res_model", "opt_seg_model", "sch_seg_model", "train_dataloader"]

        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)

    def train(self):
        self.loss_records = dict(PIX=[], TDP=[], SEG=[])
        self.on_training_start()
        while self.global_step < self.max_steps:
            pbar = self.make_pbar(total=len(self.train_dataloader))
            for batch in self.train_dataloader:
                self.prepare_batch_inputs(batch, transform=True)
                bs = self.gt.size(0)
                
                with self.accelerator.autocast():
                    self.res_model.train()
                    self.seg_model.eval().requires_grad_(False)

                    res = self.res_model(self.lq)
                    loss_pix = F.l1_loss(res, self.gt) * self.cfg.train.weight.pix

                    _, feat_gt = self.seg_model(self.gt, return_feat=True)
                    _, feat_res = self.seg_model(res, return_feat=True)
                    target_layers = self.get_target_layers()
                    loss_tdp = sum(F.l1_loss(feat_res[k], feat_gt[k]) for k in target_layers) / len(target_layers)
                    loss_tdp *= self.cfg.train.weight.tdp

                self.opt_res_model.zero_grad()
                self.accelerator.backward(loss_pix + loss_tdp)
                self.opt_res_model.step()
                self.sch_res_model.step()

                with self.accelerator.autocast():
                    self.res_model.eval()
                    self.seg_model.train().requires_grad_(True)

                    with torch.no_grad():
                        res = self.res_model(self.lq)
                    mask = (torch.randn(bs,1,8,8)).bernoulli_(p=0.5)
                    mask = F.interpolate(mask, size=self.gt.shape[2:], mode='nearest').to(self.device)
                    cqmix = res * mask + self.gt * (1-mask)

                    img_sr4ir = torch.cat((res, self.gt, cqmix), dim=0)
                    mask_sr4ir = self.mask.repeat(3,1,1)
                    pred = self.seg_model(img_sr4ir)
                    loss_seg = F.cross_entropy(pred["out"], mask_sr4ir, ignore_index=255) * self.cfg.train.weight.seg

                self.opt_seg_model.zero_grad()
                self.accelerator.backward(loss_seg)
                self.opt_seg_model.step()
                self.sch_seg_model.step()

                self.global_step += 1
                self.loss_records["PIX"].append(loss_pix.item())
                self.loss_records["TDP"].append(loss_tdp.item())
                self.loss_records["SEG"].append(loss_seg.item())
                pbar.update(1)
                pbar.set_description(
                    f"Epoch: {self.epoch:04d}, Steps: {self.global_step:07d}, "
                    f"PIX: {loss_pix.item():.3f}, TDP: {loss_tdp.item():.3f}, SEG: {loss_seg.item():.3f}"
                )

                if self.global_step % self.cfg.train.log_every == 0 or (self.args.debug):
                    self.log_training_metrics()
                if self.global_step % self.cfg.train.image_every == 0 or (self.args.debug):
                    self.seg_model.eval()
                    with self.accelerator.autocast(), torch.no_grad():
                        pred = self.seg_model(res)
                    self.seg_model.train()
                    self.log_images({"pred": mask2rgb(pred["out"].argmax(1))}, N=16)
                if self.global_step % self.cfg.train.ckpt_every == 0 or (self.args.debug):
                    self.save_checkpoints()
                if self.global_step % self.cfg.val.val_every == 0 or (self.args.debug):
                    self.evaluate()

                self.accelerator.wait_for_everyone()
                if self.global_step >= self.max_steps:
                    break
            pbar.close()
            self.epoch += 1
        self.save_checkpoints(last=True)
        self.accelerator.end_training()

    @torch.no_grad()
    def evaluate(self):
        self.res_model.eval()
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
                res = self.res_model(lq_padded)[..., :self.lq.size(2), :self.lq.size(3)]
                pred = self.seg_model(res)

                feat_hq_gt = self.seg_hq_model(self.gt, return_feat=True)[1]
                feat_hq_res = self.seg_hq_model(res, return_feat=True)[1]

            mat = calculate_mat(self.mask.flatten(), pred['out'].argmax(1).flatten(), n=self.n_classes).unsqueeze(0)

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
            self.res_model.train()
            self.seg_model.train()
