from typing import List

import torch
from torch.nn import functional as F

from pipeline.classification.base import BasePipeline
from utils.classification import calculate_accuracy
from utils.common import calculate_psnr_pt


class RestorationFixedPipeline(BasePipeline):

    def init_models(self):
        self.init_res_model(train=False)
        self.init_cls_model()

    def prepare_all(self):
        attrs = ["res_model", "cls_model", "cls_hq_model", "val_dataloader"]
        if not getattr(self.args, "eval_only", False):
            attrs += ["opt_cls_model", "sch_cls_model", "train_dataloader"]

        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)

    def train(self):
        self.loss_records = dict(CLS=[])
        self.on_training_start()
        while self.global_step < self.max_steps:
            pbar = self.make_pbar(total=len(self.train_dataloader))
            for batch in self.train_dataloader:
                self.prepare_batch_inputs(batch, transform=True)
                
                with self.accelerator.autocast():
                    with torch.no_grad():
                        res = self.res_model(self.lq).contiguous()
                    pred = self.cls_model(res)
                    loss = F.cross_entropy(pred, self.label)
                    
                self.opt_cls_model.zero_grad()
                self.accelerator.backward(loss)
                self.opt_cls_model.step()
                self.sch_cls_model.step()
                
                self.global_step += 1
                self.loss_records["CLS"].append(loss.item())
                pbar.update(1)
                pbar.set_description(
                    f"Epoch: {self.epoch:04d}, Steps: {self.global_step:07d}, Loss: {loss.item():.6f}"
                )

                if self.global_step % self.cfg.train.log_every == 0 or (self.args.debug):
                    self.log_training_metrics()
                if self.global_step % self.cfg.train.image_every == 0 or (self.args.debug):
                    self.log_images(img_dict={"restored": res}, N=12)
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
        self.cls_model.eval()

        pbar = self.make_pbar(total=len(self.val_dataloader), desc="Validation", leave=False)
        acc1_list: List[torch.Tensor] = []
        tfd_list: List[torch.Tensor] = []
        psnr_list: List[torch.Tensor] = []

        for batch in self.val_dataloader:
            self.prepare_batch_inputs(batch)

            with self.accelerator.autocast():
                res = self.res_model(self.lq)
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
            self.cls_model.train()
