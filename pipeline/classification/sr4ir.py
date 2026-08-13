from typing import List

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from pipeline.classification.base import BasePipeline
from utils.classification import calculate_accuracy
from utils.common import calculate_psnr_pt


class SR4IRPipeline(BasePipeline):

    def init_models(self):
        self.init_res_model()
        self.init_cls_model()

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

            opt_config = self.cfg.train.optimizer.cls_model
            if opt_config.type.lower() == "sgd":
                optimizer = torch.optim.SGD
            elif opt_config.type.lower() == "adamw":
                optimizer = torch.optim.AdamW
            else:
                raise NotImplementedError(f"{opt_config.type} is Not supported optimizer for cls_model")

            self.cls_model_params = list(filter(lambda p: p.requires_grad, self.cls_model.parameters()))
            self.opt_cls_model = optimizer(self.cls_model_params, **opt_config.kwargs,)
            self.sch_cls_model = CosineAnnealingLR(self.opt_cls_model, T_max=self.cfg.train.train_steps, eta_min=1e-7)

    def prepare_all(self):
        attrs = ["res_model", "cls_model", "cls_hq_model", "val_dataloader"]
        if not getattr(self.args, "eval_only", False):
            attrs += ["opt_res_model", "sch_res_model", "opt_cls_model", "sch_cls_model", "train_dataloader"]

        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)

    def train(self):
        self.loss_records = dict(PIX=[], TDP=[], CLS=[])
        self.on_training_start()
        while self.global_step < self.max_steps:
            pbar = self.make_pbar(total=len(self.train_dataloader))
            for batch in self.train_dataloader:
                self.prepare_batch_inputs(batch, transform=True)
                bs = self.gt.size(0)
                
                with self.accelerator.autocast():
                    self.res_model.train()
                    self.cls_model.eval().requires_grad_(False)

                    res = self.res_model(self.lq)
                    loss_pix = F.l1_loss(res, self.gt) * self.cfg.train.weight.pix
                    _, feat_gt = self.cls_model(self.gt, return_feat=True)
                    _, feat_res = self.cls_model(res, return_feat=True)
                    target_layers = self.get_target_layers()
                    loss_tdp = sum(F.l1_loss(feat_res[k], feat_gt[k]) for k in target_layers) / len(target_layers)
                    loss_tdp *= self.cfg.train.weight.tdp

                self.opt_res_model.zero_grad()
                self.accelerator.backward(loss_pix + loss_tdp)
                self.opt_res_model.step()
                self.sch_res_model.step()

                with self.accelerator.autocast():
                    self.res_model.eval()
                    self.cls_model.train().requires_grad_(True)

                    with torch.no_grad():
                        res = self.res_model(self.lq)
                    mask = (torch.randn(bs,1,8,8)).bernoulli_(p=0.5)
                    mask = F.interpolate(mask, size=self.gt.shape[2:], mode='nearest').to(self.device)
                    cqmix = res * mask + self.gt * (1-mask)

                    img_sr4ir = torch.cat((res, self.gt, cqmix), dim=0)
                    label_sr4ir = self.label.repeat(3)
                    pred = self.cls_model(img_sr4ir)
                    loss_cls = F.cross_entropy(pred, label_sr4ir) * self.cfg.train.weight.cls

                self.opt_cls_model.zero_grad()
                self.accelerator.backward(loss_cls)
                self.opt_cls_model.step()
                self.sch_cls_model.step()

                self.global_step += 1
                self.loss_records["PIX"].append(loss_pix.item())
                self.loss_records["TDP"].append(loss_tdp.item())
                self.loss_records["CLS"].append(loss_cls.item())
                pbar.update(1)
                pbar.set_description(
                    f"Epoch: {self.epoch:04d}, Steps: {self.global_step:07d}, "
                    f"PIX: {loss_pix.item():.3f}, TDP: {loss_tdp.item():.3f}, CLS: {loss_cls.item():.3f}"
                )

                if self.global_step % self.cfg.train.log_every == 0 or (self.args.debug):
                    self.log_training_metrics()
                if self.global_step % self.cfg.train.image_every == 0 or (self.args.debug):
                    self.log_images(img_dict={"restored": res}, N=8)
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
            self.res_model.train()
            self.cls_model.train()
