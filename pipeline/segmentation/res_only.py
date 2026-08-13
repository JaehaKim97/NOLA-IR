from typing import List

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from pipeline.segmentation.base import BasePipeline
from utils.common import calculate_psnr_pt
from utils.segmentation import calculate_mat, compute_iou

class RestorationOnlyPipeline(BasePipeline):

    def init_models(self):
        self.init_res_model()
        self.init_seg_model(train=False)

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

    def prepare_all(self):
        attrs = ["res_model", "seg_model", "seg_hq_model", "val_dataloader"]
        if not getattr(self.args, "eval_only", False):
            attrs += ["opt_res_model", "sch_res_model", "train_dataloader"]

        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)

    def train(self):
        self.loss_records = dict(PIX=[])
        self.on_training_start()
        while self.global_step < self.max_steps:
            pbar = self.make_pbar(total=len(self.train_dataloader))
            for batch in self.train_dataloader:
                self.prepare_batch_inputs(batch, transform=True)
                
                with self.accelerator.autocast():
                    res = self.res_model(self.lq)
                    loss = F.l1_loss(res, self.gt) * 255.0

                self.opt_res_model.zero_grad()
                self.accelerator.backward(loss)
                self.opt_res_model.step()
                self.sch_res_model.step()

                self.global_step += 1
                self.loss_records["PIX"].append(loss.item())
                pbar.update(1)
                pbar.set_description(
                    f"Epoch: {self.epoch:04d}, Steps: {self.global_step:07d}, Loss: {loss.item():.6f}"
                )

                if self.global_step % self.cfg.train.log_every == 0 or (self.args.debug):
                    self.log_training_metrics()
                if self.global_step % self.cfg.train.ckpt_every == 0 or (self.args.debug):
                    self.save_checkpoints()
                if self.global_step % self.cfg.train.image_every == 0 or (self.args.debug):
                    self.log_images(img_dict={"restored": res}, N=12)
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
