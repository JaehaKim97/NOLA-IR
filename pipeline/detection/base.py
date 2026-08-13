import os
from typing import List

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torchvision.utils import make_grid, save_image

from pipeline.base import CorePipeline
from utils.common import calculate_psnr_pt, instantiate_from_config, load_network, suppress_stdout
from utils.detection import (
    CocoEvaluator, GroupedBatchSampler, create_aspect_ratio_groups,
    collate_fn, draw_box, get_coco_api_from_dataset, 
)


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
        self.init_det_model()

    def init_det_model(self, train=True, instantiate_only=False):
        self.det_model = instantiate_from_config(self.cfg.model.det_model)
        if instantiate_only: return self.det_model.eval().requires_grad_(False)

        load_path, strict = self.get_model_load_path("det_model")
        if load_path:
            self.det_model = load_network(self.det_model, load_path, strict=strict)
            self.log(f"DET model: Load from: {load_path}")
        else:
            self.log("DET model: Initialize from SCRATCH")

        if train:
            self.det_model.train().requires_grad_(True)
        else:
            self.det_model.eval().requires_grad_(False)
        self.model_to_be_save.append("det_model")
        self.init_det_hq_model()

    def init_det_hq_model(self):
        self.det_hq_model = instantiate_from_config(self.cfg.model.det_model)
        load_path = self.cfg.model.det_hq_model.weights
        if load_path is None:
            self.log(f"DET-HQ model is not specified; Note that TFD has no meaning.")
        elif not os.path.exists(load_path):
            raise FileNotFoundError(f"DET-HQ model weights not found: {load_path}")
        else:
            self.det_hq_model = load_network(self.det_hq_model, load_path, strict=True)
            self.log(f"DET-HQ model: Load from: {load_path}")
        self.det_hq_model.eval().requires_grad_(False)

    def get_target_layers(self):
        target_layers = self.cfg.model.det_model.get("target_layers")
        if not target_layers:
            raise ValueError("cfg.model.det_model.target_layers must be specified for TFD/TDP/HLF computation")
        return target_layers

    def init_optimizers(self):
        if not self.args.eval_only:
            opt_config = self.cfg.train.optimizer.det_model
            self.warmup_iters = opt_config.pop("warmup_iters", 500)
            if opt_config.type.lower() == "sgd":
                optimizer = torch.optim.SGD
            elif opt_config.type.lower() == "adamw":
                optimizer = torch.optim.AdamW
            else:
                raise NotImplementedError(f"{opt_config.type} is Not supported optimizer for det_model")

            self.det_model_params = list(filter(lambda p: p.requires_grad, self.det_model.parameters()))
            self.opt_det_model = optimizer(self.det_model_params, **opt_config.kwargs)
            self.sch_det_model_warmup = LinearLR(
                self.opt_det_model,
                start_factor=(1/self.warmup_iters),
                total_iters=self.warmup_iters
            )
            self.sch_det_model = CosineAnnealingLR(
                self.opt_det_model,
                T_max=(self.cfg.train.train_steps-self.warmup_iters),
                eta_min=1e-7
            )

    def init_dataset(self):
        self.dataset_format = self.cfg.dataset.val.params.get("format", "voc").lower()
        if not self.args.eval_only:
            with suppress_stdout():
                train_dataset = instantiate_from_config(self.cfg.dataset.train)
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
            group_ids = create_aspect_ratio_groups(train_dataset, k=self.cfg.dataset.aspect_ratio_group_factor, logger=self.log)
            batch_sampler = GroupedBatchSampler(train_sampler, group_ids, self.cfg.train.batch_size)
            self.train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.cfg.dataset.num_workers,
                pin_memory=True,
                collate_fn=collate_fn
            )
            self.batch_transform = instantiate_from_config(self.cfg.dataset.batch_transform)
            self.log(f"Training dataset contains {len(train_dataset):,} images from {train_dataset.root}")

        if self.args.debug:
            self.cfg.dataset.val.params.data_length = 50
        with suppress_stdout():
            val_dataset = instantiate_from_config(self.cfg.dataset.val)
        self.log("Preparing COCO API for the validation dataset. This may take some time...")
        with suppress_stdout():
            self.val_dataset_coco_api = get_coco_api_from_dataset(val_dataset)
        if self.cfg.val.batch_size < 0:
            # use one image per process
            self.cfg.val.batch_size = self.accelerator.state.num_processes
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            drop_last=False,
            batch_size=self.cfg.val.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        self.log(f"Validation dataset contains {len(val_dataset):,} images from {val_dataset.root}")

    def prepare_all(self):
        attrs = ["det_model", "det_hq_model", "val_dataloader"]
        if not self.args.eval_only:
            attrs += ["opt_det_model", "sch_det_model_warmup", "sch_det_model", "train_dataloader"]

        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)

    def list2batch(self, img_list, img_size=512):
        max_h, max_w = img_size, img_size
        img_batch = torch.Tensor().to(self.device)
        for img in img_list:
            ph, pw = max_h - img.size(1), max_w - img.size(2)
            if img.device != self.device:
                img = img.to(self.device)
            img_padded = F.pad(img.unsqueeze(0), pad=(0, pw, 0, ph))
            img_batch = torch.cat((img_batch, img_padded), dim=0)
        return img_batch

    def batch2list(self, img_batch, original_img_list):
        new_img_list = list()
        for idx, img in enumerate(original_img_list):
            new_img_list.append(img_batch[idx][:, :img.size(1), :img.size(2)])
        return new_img_list

    def prepare_batch_inputs(self, batch, transform=False):
        self.gt_list = batch["hq"]
        batch["hq"] = self.list2batch(batch["hq"])
        if transform:
            batch = self.batch_transform(batch)
            self.gt = batch["GT"].float()
            self.lq = batch["LQ"].float()
            self.lq_list = self.batch2list(self.lq, self.gt_list)
        else:
            self.gt = batch["hq"]
            self.lq_list = batch["lq"]
            self.lq = self.list2batch(self.lq_list)
        self.prompt = batch["txt"]
        self.annot = batch["annot"]
        if "filename" in batch.keys():
            self.filename = batch["filename"]

    def draw_box(self, img, annot, score_threshold=0.6, fontsize=0.7, split_acc=False):
        is_coco = (self.dataset_format == "coco")
        score_threshold = self.cfg.val.get("score_threshold", score_threshold)
        split_acc = self.cfg.val.get("split_acc", False)
        return draw_box(img, annot, is_coco, score_threshold, fontsize, split_acc)

    def log_images(self, img_dict=None, N=12):
        if img_dict is None:
            img_dict = {}
        img_dict['gt'] = self.gt
        img_dict['gt_box'] = self.list2batch(self.draw_box(self.gt_list, self.annot))
        img_dict['lq'] = self.lq
        
        if self.accelerator.is_local_main_process:
            for tag, image in img_dict.items():
                grid_image = make_grid(image[:N], nrow=4)
                self.writer.add_image("images/" + tag, grid_image, self.global_step)
                img_name = "{}_{:06d}.png".format(tag, self.global_step)
                save_image(grid_image, os.path.join(self.img_dir, img_name))

    def save_boxed_images(self, images: List[torch.Tensor], annots, filenames, subdir: str = "", split_acc=False) -> None:
        assert (self.accelerator.state.num_processes == 1), "save_box_images requires num_processes == 1"
        boxed_images = self.draw_box(images, annots, split_acc=split_acc)
        img_root = os.path.join(self.res_dir, subdir, "img") if subdir else os.path.join(self.res_dir, "img")
        box_root = os.path.join(self.res_dir, subdir, "box") if subdir else os.path.join(self.res_dir, "box")
        os.makedirs(img_root, exist_ok=True)
        os.makedirs(box_root, exist_ok=True)

        for img, boxed, p in zip(images, boxed_images, filenames):
            name = os.path.splitext(os.path.basename(p))[0] + ".png"
            img_path = os.path.join(img_root, name)
            box_path = os.path.join(box_root, name)
            save_image(img, img_path)
            save_image(boxed, box_path)

    def train(self):
        self.loss_records = dict(DET=[])
        self.on_training_start()
        while self.global_step < self.max_steps:
            pbar = self.make_pbar(total=len(self.train_dataloader))
            for batch in self.train_dataloader:
                self.prepare_batch_inputs(batch, transform=True)
                inp_list = self.gt_list if self.cfg.train.get("use_gt", False) else self.lq_list

                with self.accelerator.autocast():
                    _, loss_dict = self.det_model(inp_list, self.annot)
                    loss = sum(loss_value for loss_value in loss_dict.values())

                self.opt_det_model.zero_grad()
                self.accelerator.backward(loss)
                self.opt_det_model.step()
                if self.global_step < self.warmup_iters:
                    self.sch_det_model_warmup.step()
                else:
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
                        pred, _ = self.det_model(inp_list)
                    self.det_model.train()
                    img_dict={"pred_box": self.list2batch(self.draw_box(inp_list, pred))}
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
            inp_list = self.gt_list if self.cfg.val.get("use_gt", False) else self.batch2list(self.lq, self.gt_list)
            assert len(inp_list) == 1

            with self.accelerator.autocast():
                pred, _ = self.det_model(inp_list)

                feat_hq_inp = self.det_hq_model(inp_list, return_feat=True)[2]
                feat_hq_gt = self.det_hq_model(self.gt_list, return_feat=True)[2]

            coco_evaluator.update({a["image_id"]: p for a, p in zip(self.annot, pred)})
            tfd = self.calculate_tfd(feat_hq_inp, feat_hq_gt)
            psnr = calculate_psnr_pt(inp_list[0].unsqueeze(0), self.gt_list[0].unsqueeze(0), crop_border=0)

            tfd, psnr = self.accelerator.gather_for_metrics((tfd, psnr))
            if self.accelerator.is_local_main_process:
                tfd_list += [0.0] * len(tfd) if self.cfg.val.get("use_gt", False) else tfd.tolist()
                psnr_list += psnr.tolist()

            if self.args.save_img:
                self.save_boxed_images(self.gt_list, self.annot, self.filename, subdir="gt", split_acc=False)
                self.save_boxed_images(inp_list, pred, self.filename, subdir="pred")

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
