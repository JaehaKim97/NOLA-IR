import os

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from torchvision.utils import save_image

from pipeline.detection.base import BasePipeline
from utils.common import Logger, copy_opt_file, instantiate_from_config, suppress_stdout
from utils.detection import collate_fn


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
        
        if accelerator.is_main_process:
            os.makedirs(self.exp_dir, exist_ok=True)
            os.makedirs(self.img_dir, exist_ok=True)

        self.log = Logger(__name__, self.exp_dir, accelerator, logger_name="logger.log")
        copy_opt_file(self.args.config, self.exp_dir)

        self.accelerator = accelerator
        self.device = accelerator.device

    def init_dataset(self):
        with suppress_stdout():
            val_dataset = instantiate_from_config(self.cfg.dataset.val)
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.cfg.val.batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        self.batch_transform = instantiate_from_config(self.cfg.dataset.batch_transform)
        self.log(f"Validation dataset contains {len(val_dataset):,} images from {val_dataset.root}")

    def prepare_all(self):
        self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

    def prepare_batch_inputs(self, batch):
        self.gt_list = batch["hq"]
        batch["hq"] = self.list2batch(batch["hq"], img_size=512)
        batch = self.batch_transform(batch)
        self.gt = batch["GT"].float()
        self.lq = batch["LQ"].float()
        self.lq_list = self.batch2list(self.lq, self.gt_list)
        self.prompt = batch["txt"]
        self.annot = batch["annot"]
        self.filename = batch["filename"]

    def train(self):
        self.data_generate()

    def evaluate(self):
        self.data_generate()

    def save_list_images(self, images: torch.Tensor, rel_paths, subdir: str) -> None:
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

            self.save_list_images(self.gt_list, self.filename, 'gt')
            self.save_list_images(self.lq_list, self.filename, 'lq')

            pbar.update(1)
        pbar.close()

        self.log("Data generation complete")
        self.accelerator.end_training()
