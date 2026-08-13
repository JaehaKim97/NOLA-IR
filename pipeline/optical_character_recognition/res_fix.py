from typing import List

import numpy as np
import torch

from pipeline.optical_character_recognition.base import BasePipeline
from utils.common import calculate_psnr_pt, log_txt_as_img
from utils.optical_character_recognition import calculate_char_accuracy


class RestorationFixedPipeline(BasePipeline):

    def init_models(self):
        self.init_converter()
        self.init_res_model(train=False)
        self.init_ocr_model()

    def prepare_all(self):
        attrs = ["res_model", "ocr_model", "val_dataloader"]
        if not getattr(self.args, "eval_only", False):
            attrs += ["opt_ocr_model", "sch_ocr_model", "train_dataloader"]

        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)

    def train(self):
        self.loss_records = dict(OCR=[])
        self.on_training_start()
        while self.global_step < self.max_steps:
            pbar = self.make_pbar(total=len(self.train_dataloader))
            for batch in self.train_dataloader:
                self.prepare_batch_inputs(batch, transform=True)
                assert self.gt.size(0) % (8 * 2) == 0, "Batch size should be divisible by 8(row) * 2(col) for tiling"
                # bh = self.gt.size(0) // 2

                with self.accelerator.autocast():
                    with torch.no_grad():
                        lq_tile = self.img2tile(self.lq)
                        res_tile = self.res_model(lq_tile)
                        res = self.tile2img(res_tile)
                    pred = self.ocr_model(res, self.text)
                    # pred = self.ocr_model(torch.cat((res[:bh], self.gt[bh:]), dim=0), self.text)
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
                        pred = self.ocr_model(res, is_train=False)
                        pred_label = self.convert2label(pred, batch_size=pred.size(0))
                    img_dict = {"restored": res, "pred_label": log_txt_as_img((128, 32), pred_label)}
                    self.log_images(img_dict, N=48)
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
            B = self.lq.size(0)
            if B % 16 != 0:
                self.lq = torch.cat([self.lq, self.lq[-1:].expand(((-B % 16), *self.lq.shape[1:]))], dim=0)

            with self.accelerator.autocast():
                lq_tile = self.img2tile(self.lq)
                res_tile = self.res_model(lq_tile)
                res = self.tile2img(res_tile)[:B]
                pred = self.ocr_model(res, is_train=False)
                pred_label = self.convert2label(pred, batch_size=B)
            
            word_correct = torch.Tensor((np.array(pred_label) == np.array(self.label))).to(self.device)
            char_correct = torch.tensor(calculate_char_accuracy(pred_label, self.label)).to(self.device)
            
            gt, res, word_correct, char_correct = self.accelerator.gather_for_metrics(
                (self.gt, res, word_correct, char_correct)
            )
            if self.accelerator.is_local_main_process:
                word_correct_list += word_correct.tolist()
                char_correct_list += char_correct.tolist()
                psnr_list += calculate_psnr_pt(res, gt, crop_border=0).detach().cpu().float().tolist()

            if self.args.save_img:
                self.save_batch_images(res, self.filename, self.label, pred_label, flatten=True)

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
