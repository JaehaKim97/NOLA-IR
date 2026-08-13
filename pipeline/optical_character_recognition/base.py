import os
from typing import List

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid, save_image

from pipeline.base import CorePipeline
from utils.common import calculate_psnr_pt, instantiate_from_config, load_network, log_txt_as_img
from utils.optical_character_recognition import calculate_char_accuracy, LabelConverter


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

    def init_environment(self):
        super().init_environment()
        self.character = self.cfg.dataset.character
        self.batch_max_length = self.cfg.dataset.batch_max_length
        self.prediction_type = self.cfg.model.ocr_model.params.Prediction.lower()

    def init_models(self):
        self.init_converter()
        self.init_ocr_model()

    def init_converter(self):
        self.converter = LabelConverter(self.character, self.batch_max_length, self.prediction_type)
        self.converter.to(self.device)

    def init_ocr_model(self, train=True, instantiate_only=False):
        self.cfg.model.ocr_model.params.num_class = len(self.converter.character)
        self.cfg.model.ocr_model.params.batch_max_length = self.batch_max_length

        self.ocr_model = instantiate_from_config(self.cfg.model.ocr_model)
        if instantiate_only: return self.ocr_model.eval().requires_grad_(False)

        load_path, strict = self.get_model_load_path("ocr_model")
        if load_path:
            self.ocr_model = load_network(self.ocr_model, load_path, strict=strict)
            self.log(f"OCR model: Load from: {load_path}")
        else:
            self.log("OCR model: Initialize from SCRATCH")

        if train:
            self.ocr_model.train().requires_grad_(True)
        else:
            self.ocr_model.eval().requires_grad_(False)
        self.model_to_be_save.append("ocr_model")
        # self.init_ocr_hq_model()

    def init_ocr_hq_model(self):
        self.ocr_hq_model = instantiate_from_config(self.cfg.model.ocr_model)
        load_path = self.cfg.model.ocr_hq_model.weights
        if load_path is None:
            self.log(f"OCR-HQ model is not specified")
        elif not os.path.exists(load_path):
            raise FileNotFoundError(f"OCR-HQ model weights not found: {load_path}")
        else:
            self.ocr_hq_model = load_network(self.ocr_hq_model, load_path, strict=True)
            self.log(f"OCR-HQ model: Load from: {load_path}")
        self.ocr_hq_model.eval().requires_grad_(False)

    def get_target_layers(self):
        target_layers = self.cfg.model.ocr_model.get("target_layers")
        if not target_layers:
            raise ValueError("cfg.model.ocr_model.target_layers must be specified for TFD/TDP/HLF computation")
        return target_layers

    def init_optimizers(self):
        if not self.args.eval_only:
            opt_config = self.cfg.train.optimizer.ocr_model
            if opt_config.type.lower() == "sgd":
                optimizer = torch.optim.SGD
            elif opt_config.type.lower() == "adam":
                optimizer = torch.optim.Adam
            else:
                raise NotImplementedError(f"{opt_config.type} is Not supported optimizer for ocr_model")

            self.ocr_model_params = list(filter(lambda p: p.requires_grad, self.ocr_model.parameters()))
            self.opt_ocr_model = optimizer(self.ocr_model_params, **opt_config.kwargs)
            self.sch_ocr_model = CosineAnnealingLR(self.opt_ocr_model, T_max=self.cfg.train.train_steps, eta_min=1e-7)

    def init_dataset(self):
        if not self.args.eval_only:
            self.cfg.dataset.train.params.batch_max_length = self.batch_max_length
            train_dataset = instantiate_from_config(self.cfg.dataset.train)
            self.train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                drop_last=True,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.dataset.num_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
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
        attrs = ["ocr_model", "val_dataloader"]
        if not self.args.eval_only:
            attrs += ["opt_ocr_model", "sch_ocr_model", "train_dataloader"]

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

        self.text, self.length = self.converter.encode(self.label)

    def img2tile(self, img, row=8, col=2):
        assert (img.size(0) % (row*col)) == 0
        B, C, H, W = img.shape
        tile = img.view(B//(row*col), row, col, C, H, W)
        tile = tile.permute(0, 3, 1, 4, 2, 5).reshape(B//(row*col), C, row*H, col*W).contiguous()
        return tile

    def tile2img(self, tile, row=8, col=2):
        B, C, H, W = tile.shape
        img = tile.view(B, C, row, H//row, col, W//col)
        img = img.permute(0, 2, 4, 1, 3, 5).reshape(B*(row*col), C, H//row, W//col).contiguous()
        return img

    def calculate_ocr_loss(self, pred, text=None, length=None):
        pred_size = torch.IntTensor([pred.size(1)] * pred.size(0))
        if text is None:
            text = self.text
        if length is None:
            length = self.length

        if self.prediction_type == "ctc":
            pred_log_softmax = pred.log_softmax(2).permute(1, 0, 2)
            return F.ctc_loss(pred_log_softmax, text, pred_size, length, zero_infinity=True)
        elif self.prediction_type == "attn":
            return F.cross_entropy(
                pred.view(-1, pred.shape[-1]), text[:, 1:].contiguous().view(-1), ignore_index=0
            )

    def log_images(self, img_dict=None, N=12):
        if img_dict is None:
            img_dict = {}
        img_dict['gt'] = self.gt
        img_dict['label'] = log_txt_as_img((128, 32), self.label)
        img_dict['lq'] = self.lq

        if self.accelerator.is_local_main_process:
            for tag, image in img_dict.items():
                grid_image = make_grid(image[:N], nrow=4)
                self.writer.add_image("images/" + tag, grid_image, self.global_step)
                img_name = "{}_{:06d}.png".format(tag, self.global_step)
                save_image(grid_image, os.path.join(self.img_dir, img_name))

    def save_batch_images(self, images: torch.Tensor, rel_paths, gt, pred, flatten=False, exist_gt=True) -> None:
        for i, p in enumerate(rel_paths):
            tail = os.path.join(*p.split("/")[-1:])
            if flatten:
                tail = tail.replace("/", "_")
            if exist_gt:
                res = f"_[GT:{gt[i]}-Pred:{pred[i]}]"
            else:
                res = f"_[Pred:{pred[i]}]"
            out_path = os.path.splitext(os.path.join(self.res_dir, tail))[0] + res + ".png"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            save_image(images[i], out_path)

    def convert2label(self, pred, batch_size=None):
        if batch_size is None:
            batch_size = self.lq.size(0)

        if self.prediction_type == "ctc":
            pred_size = torch.IntTensor([pred.size(1)] * batch_size)
            _, pred_index = pred.max(2)
            pred_label = self.converter.decode(pred_index.data, pred_size.data)
        elif self.prediction_type == "attn":
            text_for_loss, length_for_loss = self.converter.encode(self.label)
            pred = pred[:, :text_for_loss.shape[1] - 1, :]
            _, pred_index = pred.max(2)

            # select max probabilty (greedy decoding) then decode index to character
            length_for_pred = torch.IntTensor([self.batch_max_length] * batch_size).to(self.device)
            pred_label = self.converter.decode(pred_index, length_for_pred)
            pred_label = [s.split('[s]')[0] for s in pred_label]

        return pred_label

    def train(self):
        self.loss_records = dict(OCR=[])
        self.on_training_start()
        while self.global_step < self.max_steps:
            pbar = self.make_pbar(total=len(self.train_dataloader))
            for batch in self.train_dataloader:
                self.prepare_batch_inputs(batch, transform=True)
                inp = self.gt if self.cfg.train.get("use_gt", False) else self.lq

                with self.accelerator.autocast():
                    pred = self.ocr_model(inp, self.text)
                    loss = self.calculate_ocr_loss(pred)

                self.opt_ocr_model.zero_grad()
                self.accelerator.backward(loss)
                self.opt_ocr_model.step()
                self.sch_ocr_model.step()

                self.global_step += 1
                self.loss_records["OCR"].append(loss.item())
                pbar.update(1)
                pbar.set_description(
                    f"Epoch: {self.epoch:04d}, Steps: {self.global_step:07d}, Loss: {loss.item():.6f}"
                )

                if self.global_step % self.cfg.train.log_every == 0 or (self.args.debug):
                    self.log_training_metrics()
                if self.global_step % self.cfg.train.image_every == 0 or (self.args.debug):
                    self.ocr_model.eval()
                    with self.accelerator.autocast(), torch.no_grad():
                        pred = self.ocr_model(inp, is_train=False)
                        pred_label = self.convert2label(pred)
                    self.log_images({"pred_label": log_txt_as_img((128, 32), pred_label)}, N=64)
                    self.ocr_model.train()
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
        self.ocr_model.eval()

        pbar = self.make_pbar(total=len(self.val_dataloader), desc="Validation", leave=False)
        word_correct_list: List[torch.Tensor] = []
        char_correct_list: List[torch.Tensor] = []
        psnr_list: List[torch.Tensor] = []

        for batch in self.val_dataloader:
            self.prepare_batch_inputs(batch)
            inp = self.gt if self.cfg.val.get("use_gt", False) else self.lq

            with self.accelerator.autocast():
                pred = self.ocr_model(inp, is_train=False)
                pred_label = self.convert2label(pred)
            
            word_correct = torch.Tensor((np.array(pred_label) == np.array(self.label))).to(self.device)
            char_correct = torch.tensor(calculate_char_accuracy(pred_label, self.label)).to(self.device)
            
            gt, inp, word_correct, char_correct = self.accelerator.gather_for_metrics(
                (self.gt, inp, word_correct, char_correct)
            )
            if self.accelerator.is_local_main_process:
                psnr_list += calculate_psnr_pt(inp, gt, crop_border=0).detach().cpu().float().tolist()
                word_correct_list += word_correct.tolist()
                char_correct_list += char_correct.tolist()

            if self.args.save_img:
                self.save_batch_images(inp, self.filename, self.label, pred_label, flatten=True)

            pbar.update(1)
        pbar.close()

        if self.accelerator.is_local_main_process:
            avg_word_acc = torch.tensor(word_correct_list).mean().item()
            avg_char_acc = torch.tensor(char_correct_list).mean().item()
            avg_psnr = torch.tensor(psnr_list).mean().item()
            for tag, val in [("val/word_acc", avg_word_acc), ("val/char_acc", avg_char_acc), ("val/psnr", avg_psnr)]:
                self.log(f"{tag}: {val:.4f}")
                if not self.args.eval_only:
                    self.writer.add_scalar(tag, val, self.global_step)

        self.accelerator.wait_for_everyone()
        if self.args.eval_only:
            self.accelerator.end_training()
        else:
            self.ocr_model.train()
