import os
from typing import List

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid, save_image

from pipeline.base import CorePipeline
from utils.classification import calculate_accuracy
from utils.common import calculate_psnr_pt, instantiate_from_config, load_network


class BasePipeline(CorePipeline):

    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.init_environment()
        self.init_models()
        self.summary_models()
        self.init_optimizers()
        self.init_dataset()
        self.prepare_all()

    def init_models(self):
        self.init_cls_model()

    def init_cls_model(self, train=True):
        self.cls_model = instantiate_from_config(self.cfg.model.cls_model)
        load_path, strict = self.get_model_load_path("cls_model")
        if load_path:
            self.cls_model = load_network(self.cls_model, load_path, strict=strict)
            self.log(f"CLS model: Load from: {load_path}")
        else:
            self.log("CLS model: Initialize from SCRATCH")

        if train:
            self.cls_model.train().requires_grad_(True)
        else:
            self.cls_model.eval().requires_grad_(False)
        self.model_to_be_save.append("cls_model")
        self.init_cls_hq_model()

    def init_cls_hq_model(self):
        self.cls_hq_model = instantiate_from_config(self.cfg.model.cls_model)
        load_path = self.cfg.model.cls_hq_model.weights
        if load_path is None:
            self.log("CLS-HQ model is not specified; Note that TFD has no meaning")
        elif not os.path.exists(load_path):
            raise FileNotFoundError(f"CLS-HQ model weights not found: {load_path}")
        else:
            self.cls_hq_model = load_network(self.cls_hq_model, load_path, strict=True)
            self.log(f"CLS-HQ model: Load from: {load_path}")
        self.cls_hq_model.eval().requires_grad_(False)

    def get_target_layers(self):
        target_layers = self.cfg.model.cls_model.get("target_layers")
        if not target_layers:
            raise ValueError("cfg.model.cls_model.target_layers must be specified for TFD/TDP/HLF computation")
        return target_layers

    def init_optimizers(self):
        if not self.args.eval_only:
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

    def init_dataset(self):
        if not self.args.eval_only:
            train_dataset = instantiate_from_config(self.cfg.dataset.train)
            self.train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                drop_last=True,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.dataset.num_workers,
            )
            self.batch_transform = instantiate_from_config(self.cfg.dataset.batch_transform)
            self.log(f"Training dataset contains {len(train_dataset):,} images from {train_dataset.root}")

        val_dataset = instantiate_from_config(self.cfg.dataset.val)
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            drop_last=False,
            batch_size=self.cfg.val.batch_size,
            num_workers=self.cfg.dataset.num_workers,
        )
        self.log(f"Validation dataset contains {len(val_dataset):,} images from {val_dataset.root}")

    def prepare_all(self):
        attrs = ["cls_model", "cls_hq_model", "val_dataloader"]
        if not self.args.eval_only:
            attrs += ["opt_cls_model", "sch_cls_model", "train_dataloader"]

        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)

    def prepare_batch_inputs(self, batch, transform=False):
        if transform:
            batch = self.batch_transform(batch)
            self.gt = batch["GT"].float()
            self.lq = batch["LQ"].float()
        else:
            self.gt = batch["hq"]
            self.lq = batch["lq"]
        self.prompt = batch["txt"]
        self.label = batch["label"]
        if "filename" in batch.keys():
            self.filename = batch["filename"]

    def log_images(self, img_dict=None, N=12):
        if img_dict is None:
            img_dict = {}
        img_dict['gt'] = self.gt
        img_dict['lq'] = self.lq
        
        if self.accelerator.is_local_main_process:
            for tag, image in img_dict.items():
                grid_image = make_grid(image[:N], nrow=4)
                self.writer.add_image("images/" + tag, grid_image, self.global_step)
                img_name = "{}_{:06d}.png".format(tag, self.global_step)
                save_image(grid_image, os.path.join(self.img_dir, img_name))

    def save_batch_images(self, images: torch.Tensor, rel_paths, gt, pred, flatten=False) -> None:
        for i, p in enumerate(rel_paths):
            tail = os.path.join(*p.split("/")[-2:])
            if flatten:
                tail = tail.replace("/", "_")
            res = "_[GT{:03d}-Pred{:03d}]".format(gt[i]+1, pred.argmax(1)[i]+1)
            out_path = os.path.splitext(os.path.join(self.res_dir, tail))[0] + res + ".png"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            save_image(images[i], out_path)

    def train(self):
        self.loss_records = dict(CLS=[])
        self.on_training_start()
        while self.global_step < self.max_steps:
            pbar = self.make_pbar(total=len(self.train_dataloader))
            for batch in self.train_dataloader:
                self.prepare_batch_inputs(batch, transform=True)
                inp = self.gt if self.cfg.train.get("use_gt", False) else self.lq

                with self.accelerator.autocast():
                    pred = self.cls_model(inp)
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
                # self.global_step % self.cfg.train.image_every == 0 or (self.args.debug)
                #     self.log_images()
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
            inp = self.gt if self.cfg.val.get("use_gt", False) else self.lq

            with self.accelerator.autocast():
                pred = self.cls_model(inp)

                feat_hq_inp = self.cls_hq_model(inp, return_feat=True)[1]
                feat_hq_gt = self.cls_hq_model(self.gt, return_feat=True)[1]

            tfd = self.calculate_tfd(feat_hq_inp, feat_hq_gt)
            psnr = calculate_psnr_pt(inp, self.gt, crop_border=0)

            pred, label, tfd, psnr = self.accelerator.gather_for_metrics((pred, self.label, tfd, psnr))
            if self.accelerator.is_local_main_process:
                acc1_list += [calculate_accuracy(pred, label, topk=(1, 5))[0]] * pred.size(0)
                tfd_list += [0.0] * len(tfd) if self.cfg.val.get("use_gt", False) else tfd.tolist()
                psnr_list += psnr.tolist()

            if self.args.save_img:
                self.save_batch_images(inp, self.filename, self.label, pred, flatten=True)

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
