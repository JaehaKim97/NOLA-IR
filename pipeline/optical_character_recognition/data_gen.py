import json
import os
import pathlib

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from torchvision.utils import save_image

from pipeline.optical_character_recognition.base import BasePipeline
from utils.common import Logger, copy_opt_file, instantiate_from_config


class DataGenerationPipeline(BasePipeline):

    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.init_environment()
        self.init_dataset()
        self.prepare_all()

    def init_environment(self):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        set_seed(self.cfg.seed)

        self.exp_dir = self.cfg.exp_dir
        self.img_dir = os.path.join(self.exp_dir, "images")

        accelerator = Accelerator(
            dataloader_config=DataLoaderConfiguration(split_batches=True),
            mixed_precision=self.cfg.precision,
        )

        self.save_json = self.cfg.val.get("save_json", False)
        if accelerator.is_main_process:
            os.makedirs(self.exp_dir, exist_ok=True)
            os.makedirs(self.img_dir, exist_ok=True)
            if self.save_json:
                json_path = os.path.join(self.img_dir, "data.jsonl")
                self.json = pathlib.Path(json_path).open("w", encoding="utf-8")

        self.log = Logger(__name__, self.exp_dir, accelerator, logger_name="logger.log")
        copy_opt_file(self.args.config, self.exp_dir)

        self.accelerator = accelerator
        self.device = accelerator.device

        self.character = self.cfg.dataset.character
        self.batch_max_length = self.cfg.dataset.batch_max_length

    def init_dataset(self):
        self.cfg.dataset.val.params.batch_max_length = self.batch_max_length
        val_dataset = instantiate_from_config(self.cfg.dataset.val)
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            drop_last=False,
            batch_size=self.cfg.val.batch_size,
            num_workers=self.cfg.dataset.num_workers,
        )
        self.batch_transform = instantiate_from_config(self.cfg.dataset.batch_transform)
        self.log(f"Validation dataset contains {len(val_dataset):,} images from {val_dataset.root}")

    def prepare_all(self):
        self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

    def prepare_batch_inputs(self, batch):
        batch = self.batch_transform(batch)
        self.gt = batch["GT"].float()
        self.lq = batch["LQ"].float()
        self.prompt = batch["txt"]
        self.label = batch["label"]
        self.filename = batch["filename"]

    def train(self):
        self.data_generate()

    def evaluate(self):
        self.data_generate()

    def save_batch_images(self, images: torch.Tensor, rel_paths, subdir: str) -> None:
        out_root = os.path.join(self.img_dir, subdir)

        for i, p in enumerate(rel_paths):
            tail = p.split("/")[-1:]
            out_path = os.path.splitext(os.path.join(out_root, *tail))[0] + ".png"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            save_image(images[i], out_path)

    @torch.no_grad()
    def data_generate(self):
        pbar = self.make_pbar(total=len(self.val_dataloader), desc="Data Generation", leave=False)

        for batch in self.val_dataloader:
            self.prepare_batch_inputs(batch)

            self.save_batch_images(self.gt, self.filename, 'gt')
            self.save_batch_images(self.lq, self.filename, 'lq')

            if self.save_json:
                for filename, label in zip(self.filename, self.label):
                    pair = {"image": filename, "label": label}
                    self.json.write(json.dumps(pair, ensure_ascii=False) + "\n")

            pbar.update(1)
        pbar.close()

        if self.save_json:
            self.json.close()

        self.log("Data generation complete")
        self.accelerator.end_training()
