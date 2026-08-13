import math
import os
from typing import List

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid, save_image

from pipeline.base import CorePipeline
from utils.common import calculate_psnr_pt, instantiate_from_config, load_network, suppress_stdout
from utils.segmentation import calculate_mat, compute_iou, mask2rgb


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
        self.init_seg_model()

    def init_seg_model(self, train=True):
        self.seg_model = instantiate_from_config(self.cfg.model.seg_model)
        load_path, strict = self.get_model_load_path("seg_model")
        if load_path:
            self.seg_model = load_network(self.seg_model, load_path, strict=strict)
            self.log(f"SEG model: Load from: {load_path}")
        else:
            self.log("SEG model: Initialize from SCRATCH")

        if train:
            self.seg_model.train().requires_grad_(True)
        else:
            self.seg_model.eval().requires_grad_(False)
        self.model_to_be_save.append("seg_model")
        self.init_seg_hq_model()

    def init_seg_hq_model(self):
        self.seg_hq_model = instantiate_from_config(self.cfg.model.seg_model)
        load_path = self.cfg.model.seg_hq_model.weights
        if load_path is None:
            self.log(f"SEG-HQ model is not specified; Note that TFD has no meaning")
        elif not os.path.exists(load_path):
            raise FileNotFoundError(f"SEG-HQ model weights not found: {load_path}")
        else:
            self.seg_hq_model = load_network(self.seg_hq_model, load_path, strict=True)
            self.log(f"SEG-HQ model: Load from: {load_path}")
        self.seg_hq_model.eval().requires_grad_(False)

    def get_target_layers(self):
        target_layers = self.cfg.model.seg_model.get("target_layers")
        if not target_layers:
            raise ValueError("cfg.model.seg_model.target_layers must be specified for TFD/TDP/HLF computation")
        return target_layers

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

    def init_dataset(self):
        if not self.args.eval_only:
            with suppress_stdout():
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

        with suppress_stdout():
            val_dataset = instantiate_from_config(self.cfg.dataset.val)
        self.n_classes = val_dataset.n_classes
        if self.cfg.val.batch_size < 0:
            # use one image per process
            self.cfg.val.batch_size = self.accelerator.state.num_processes
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            drop_last=False,
            batch_size=self.cfg.val.batch_size,
            num_workers=self.cfg.dataset.num_workers,
        )
        self.log(f"Validation dataset contains {len(val_dataset):,} images from {val_dataset.root}")

    def prepare_all(self):
        attrs = ["seg_model", "seg_hq_model", "val_dataloader"]
        if not self.args.eval_only:
            attrs += ["opt_seg_model", "sch_seg_model", "train_dataloader"]

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
        self.mask = batch["mask"]
        if "filename" in batch.keys():
            self.filename = batch["filename"]

    def log_images(self, img_dict=None, N=12):
        if img_dict is None:
            img_dict = {}
        img_dict['gt'] = self.gt
        img_dict["gt_mask"] = mask2rgb(self.mask)
        img_dict['lq'] = self.lq
        
        if self.accelerator.is_local_main_process:
            for tag, image in img_dict.items():
                grid_image = make_grid(image[:N], nrow=4)
                self.writer.add_image("images/" + tag, grid_image, self.global_step)
                img_name = "{}_{:06d}.png".format(tag, self.global_step)
                save_image(grid_image, os.path.join(self.img_dir, img_name))

    def pad_image(self, img, multiple=None, img_size=None, mode='constant'):
        h, w = img.shape[2:]

        if multiple is not None:
            ph = math.ceil(h / multiple) * multiple - h
            pw = math.ceil(w / multiple) * multiple - w
        elif img_size is not None:
            ph = img_size - h
            pw = img_size - w
        else:
            raise ValueError("Either 'multiple' or 'img_size' must be specified")

        return F.pad(img, pad=(0, pw, 0, ph), mode=mode)

    def save_masked_image(self, image, mask, filenames, subdir: str = "") -> None:
        assert (self.accelerator.state.num_processes == 1), "save_masked_image requires num_processes == 1"
        masked_image = mask2rgb(mask)
        img_root = os.path.join(self.res_dir, subdir, "img") if subdir else os.path.join(self.res_dir, "img")
        mask_root = os.path.join(self.res_dir, subdir, "mask") if subdir else os.path.join(self.res_dir, "mask")
        name = os.path.splitext(os.path.basename(filenames[0]))[0] + ".png"
        
        for root, image in [(img_root, image), (mask_root, masked_image)]:
            os.makedirs(root, exist_ok=True)
            img_path = os.path.join(root, name)
            save_image(image, img_path)

    def train(self):
        self.loss_records = dict(SEG=[])
        self.on_training_start()
        while self.global_step < self.max_steps:
            pbar = self.make_pbar(total=len(self.train_dataloader))
            for batch in self.train_dataloader:
                self.prepare_batch_inputs(batch, transform=True)
                inp = self.gt if self.cfg.train.get("use_gt", False) else self.lq

                with self.accelerator.autocast():
                    pred = self.seg_model(inp)
                    loss = F.cross_entropy(pred["out"], self.mask, ignore_index=255)

                self.opt_seg_model.zero_grad()
                self.accelerator.backward(loss)
                self.opt_seg_model.step()
                self.sch_seg_model.step()

                self.global_step += 1
                self.loss_records["SEG"].append(loss.item())
                pbar.update(1)
                pbar.set_description(
                    f"Epoch: {self.epoch:04d}, Steps: {self.global_step:07d}, Loss: {loss.item():.6f}"
                )

                if self.global_step % self.cfg.train.log_every == 0 or (self.args.debug):
                    self.log_training_metrics()
                if self.global_step % self.cfg.train.image_every == 0 or (self.args.debug):
                    self.seg_model.eval()
                    with self.accelerator.autocast(), torch.no_grad():
                        pred = self.seg_model(inp)
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
        self.seg_model.eval()

        pbar = self.make_pbar(total=len(self.val_dataloader), desc="Validation", leave=False)
        confmat = torch.zeros((self.n_classes, self.n_classes), device=self.device)
        tfd_list: List[torch.Tensor] = []
        psnr_list: List[torch.Tensor] = []

        for batch in self.val_dataloader:
            self.prepare_batch_inputs(batch)
            inp = self.gt if self.cfg.val.get("use_gt", False) else self.lq
            assert inp.size(0) == 1

            with self.accelerator.autocast():
                pred = self.seg_model(inp)

                feat_hq_inp = self.seg_hq_model(inp, return_feat=True)[1]
                feat_hq_gt = self.seg_hq_model(self.gt, return_feat=True)[1]

            mat = calculate_mat(self.mask.flatten(), pred['out'].argmax(1).flatten(), n=self.n_classes).unsqueeze(0)
            tfd = self.calculate_tfd(feat_hq_inp, feat_hq_gt)
            psnr = calculate_psnr_pt(inp, self.gt, crop_border=0)

            mat, tfd, psnr = self.accelerator.gather_for_metrics((mat, tfd, psnr))
            if self.accelerator.is_local_main_process:
                confmat += mat.sum(0)
                tfd_list += [0.0] * len(tfd) if self.cfg.val.get("use_gt", False) else tfd.tolist()
                psnr_list += psnr.tolist()

            if self.args.save_img:
                self.save_masked_image(self.gt, self.mask, self.filename, subdir="gt")
                self.save_masked_image(inp, pred["out"].argmax(1), self.filename, subdir="pred")

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
            self.seg_model.train()
