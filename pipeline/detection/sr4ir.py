from typing import List

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from pipeline.detection.base import BasePipeline
from utils.common import calculate_psnr_pt, suppress_stdout
from utils.detection import CocoEvaluator


class SR4IRPipeline(BasePipeline):

    def init_models(self):
        self.init_res_model()
        self.init_det_model()

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

            opt_config = self.cfg.train.optimizer.det_model
            if opt_config.type.lower() == "sgd":
                optimizer = torch.optim.SGD
            elif opt_config.type.lower() == "adamw":
                optimizer = torch.optim.AdamW
            else:
                raise NotImplementedError(f"{opt_config.type} is Not supported optimizer for det_model")

            self.det_model_params = list(filter(lambda p: p.requires_grad, self.det_model.parameters()))
            self.opt_det_model = optimizer(self.det_model_params, **opt_config.kwargs,)
            self.sch_det_model = CosineAnnealingLR(self.opt_det_model, T_max=self.cfg.train.train_steps, eta_min=1e-7)

    def prepare_all(self):
        attrs = ["res_model", "det_model", "det_hq_model", "val_dataloader"]
        if not getattr(self.args, "eval_only", False):
            attrs += ["opt_res_model", "sch_res_model", "opt_det_model", "sch_det_model", "train_dataloader"]

        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)

    def train(self):
        self.loss_records = dict(PIX=[], TDP=[], DET=[])
        self.on_training_start()
        while self.global_step < self.max_steps:
            pbar = self.make_pbar(total=len(self.train_dataloader))
            for batch in self.train_dataloader:
                self.prepare_batch_inputs(batch, transform=True)
                bs = self.gt.size(0)
                
                with self.accelerator.autocast():
                    self.res_model.train()
                    self.det_model.eval().requires_grad_(False)

                    res = self.res_model(self.lq)
                    loss_pix = F.l1_loss(res, self.gt) * self.cfg.train.weight.pix

                    res_list = self.batch2list(res, self.gt_list)
                    _, _, feat_gt = self.det_model(self.gt_list, return_feat=True)
                    _, _, feat_res = self.det_model(res_list, return_feat=True)
                    target_layers = self.get_target_layers()
                    loss_tdp = sum(F.l1_loss(feat_res[k], feat_gt[k]) for k in target_layers) / len(target_layers)
                    loss_tdp *= self.cfg.train.weight.tdp

                self.opt_res_model.zero_grad()
                self.accelerator.backward(loss_pix + loss_tdp)
                self.opt_res_model.step()
                self.sch_res_model.step()

                with self.accelerator.autocast():
                    self.res_model.eval()
                    self.det_model.train().requires_grad_(True)

                    with torch.no_grad():
                        res = self.res_model(self.lq)
                        res_list = self.batch2list(res, self.gt_list)
                    mask = (torch.randn(bs,1,8,8)).bernoulli_(p=0.5)
                    mask = F.interpolate(mask, size=self.gt.shape[2:], mode='nearest').to(self.device)
                    cqmix = res * mask + self.gt * (1-mask)
                    cqmix_list = self.batch2list(cqmix, self.gt_list)

                    img_sr4ir_list = res_list + self.gt_list + cqmix_list
                    annot_sr4ir = self.annot * 3
                    _, loss_dict = self.det_model(img_sr4ir_list, annot_sr4ir)
                    loss_det = sum(loss_value for loss_value in loss_dict.values()) * self.cfg.train.weight.det

                self.opt_det_model.zero_grad()
                self.accelerator.backward(loss_det)
                self.opt_det_model.step()
                self.sch_det_model.step()

                self.global_step += 1
                self.loss_records["PIX"].append(loss_pix.item())
                self.loss_records["TDP"].append(loss_tdp.item())
                self.loss_records["DET"].append(loss_det.item())
                pbar.update(1)
                pbar.set_description(
                    f"Epoch: {self.epoch:04d}, Steps: {self.global_step:07d}, "
                    f"PIX: {loss_pix.item():.3f}, TDP: {loss_tdp.item():.3f}, DET: {loss_det.item():.3f}"
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
        self.det_model.eval()

        pbar = self.make_pbar(total=len(self.val_dataloader), desc="Validation", leave=False)
        coco_evaluator = CocoEvaluator(self.val_dataset_coco_api, ['bbox'])
        tfd_list: List[torch.Tensor] = []
        psnr_list: List[torch.Tensor] = []

        for batch in self.val_dataloader:
            self.prepare_batch_inputs(batch)

            with self.accelerator.autocast():
                res = self.res_model(self.lq)
                res_list = self.batch2list(res, self.gt_list)
                pred, _ = self.det_model(res_list)

                feat_hq_res = self.det_hq_model(res_list, return_feat=True)[2]
                feat_hq_gt = self.det_hq_model(self.gt_list, return_feat=True)[2]

            coco_evaluator.update({a["image_id"]: p for a, p in zip(self.annot, pred)})
            tfd = self.calculate_tfd(feat_hq_res, feat_hq_gt)
            psnr = calculate_psnr_pt(res_list[0].unsqueeze(0), self.gt_list[0].unsqueeze(0), crop_border=0)

            tfd, psnr = self.accelerator.gather_for_metrics((tfd, psnr))
            if self.accelerator.is_local_main_process:
                tfd_list += tfd.tolist()
                psnr_list += psnr.tolist()

            if self.args.save_img:
                self.save_boxed_images(res_list, pred, self.filename, split_acc=True)

            pbar.update(1)
        pbar.close()

        coco_evaluator.synchronize_between_processes()
        with suppress_stdout():
            coco_evaluator.accumulate()

        if self.accelerator.is_local_main_process:
            det_results = coco_evaluator.summarize(logger=self.log)
            avg_tfd = torch.tensor(tfd_list).mean().item()
            avg_psnr = torch.tensor(psnr_list).mean().item()
            for tag, val in [
                ("val/mAP@[0.5:0.95]", det_results["mAP@[0.5:0.95]"]),
                ("val/mAP@0.5", det_results["mAP@0.5"]),
                ("val/tfd", avg_tfd), ("val/psnr", avg_psnr),
            ]:
                self.log(f"{tag}: {val:.4f}")
                if not self.args.eval_only:
                    self.writer.add_scalar(tag, val, self.global_step)

        self.accelerator.wait_for_everyone()
        if self.args.eval_only:
            self.accelerator.end_training()
        else:
            self.res_model.train()
            self.det_model.train()
