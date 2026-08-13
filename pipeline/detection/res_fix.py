from typing import List

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from pipeline.detection.base import BasePipeline
from utils.common import calculate_psnr_pt, suppress_stdout
from utils.detection import CocoEvaluator


class RestorationFixedPipeline(BasePipeline):

    def init_models(self):
        self.init_res_model(train=False)
        self.init_det_model()

    def init_optimizers(self):
        if not self.args.eval_only:
            opt_config = self.cfg.train.optimizer.det_model
            if opt_config.type.lower() == "sgd":
                optimizer = torch.optim.SGD
            elif opt_config.type.lower() == "adamw":
                optimizer = torch.optim.AdamW
            else:
                raise NotImplementedError(f"{opt_config.type} is Not supported optimizer for det_model")

            self.det_model_params = list(filter(lambda p: p.requires_grad, self.det_model.parameters()))
            self.opt_det_model = optimizer(self.det_model_params, **opt_config.kwargs)
            self.sch_det_model = CosineAnnealingLR(self.opt_det_model,T_max=self.cfg.train.train_steps, eta_min=1e-7)
    
    def prepare_all(self):
        attrs = ["res_model", "det_model", "det_hq_model", "val_dataloader"]
        if not getattr(self.args, "eval_only", False):
            attrs += ["opt_det_model", "sch_det_model", "train_dataloader"]

        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)

    def train(self):
        self.loss_records = dict(DET=[])
        self.on_training_start()
        while self.global_step < self.max_steps:
            pbar = self.make_pbar(total=len(self.train_dataloader))
            for batch in self.train_dataloader:
                self.prepare_batch_inputs(batch, transform=True)

                with self.accelerator.autocast():
                    with torch.no_grad():
                        res = self.res_model(self.lq)
                    res_list = self.batch2list(res, self.gt_list)
                    _, loss_dict = self.det_model(res_list, self.annot)
                    loss = sum(loss_value for loss_value in loss_dict.values())

                self.opt_det_model.zero_grad()
                self.accelerator.backward(loss)
                self.opt_det_model.step()
                self.sch_det_model.step()

                self.global_step += 1
                self.loss_records["DET"].append(loss.item())
                pbar.update(1)
                pbar.set_description(
                    f"Epoch: {self.epoch:04d}, Steps: {self.global_step:07d}, Loss: {loss.item():.6f}"
                )

                if self.global_step % self.cfg.train.log_every == 0 or (self.args.debug):
                    self.log_training_metrics()
                if self.global_step % self.cfg.train.image_every == 0 or (self.args.debug):
                    self.det_model.eval()
                    with self.accelerator.autocast(), torch.no_grad():
                        pred, _ = self.det_model(res_list)
                    self.det_model.train()
                    img_dict={"pred_box": self.list2batch(self.draw_box(res_list, pred))}
                    self.log_images(img_dict=img_dict, N=16)
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
                self.save_boxed_images(res_list, pred, self.filename)

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
            self.det_model.train()