import os

import numpy as np
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from einops import rearrange
from glob import glob
from PIL import Image
from tqdm import tqdm

from pipeline.detection.nola_ir import NOLAIRPipeline
from utils.common import wavelet_reconstruction, pad_if_smaller, pad_to_multiples_of


class NOLAIRDemoPipeline(NOLAIRPipeline):

    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.init_environment()
        self.init_dataset()
        self.summary_config()
        self.init_models()
        self.prepare_all()

    def init_environment(self):
        self.res_dir = self.args.output
        os.makedirs(self.res_dir, exist_ok=True)

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        set_seed(self.cfg.seed)
        accelerator = Accelerator(
            dataloader_config=DataLoaderConfiguration(split_batches=True),
            mixed_precision=self.cfg.precision,
        )
        self.accelerator = accelerator
        self.device = accelerator.device
        mp = accelerator.mixed_precision
        self.weight_dtype = torch.float16 if mp == "fp16" else torch.bfloat16 if mp == "bf16" else torch.float32

    def summary_config(self):
        scale_str = f"{self.args.scale:.2f}" if self.args.scale is not None else "auto (long side 512)"
        print(f"NOLA-IR  |  task: {self.args.task}  |  device: {self.device}")
        print(f"    config   {self.args.config}")
        print(f"    weights  {self.cfg.val.weights}")
        print(f"    input    {self.args.input}  ({len(self.img_paths)} images)")
        print(f"    output   {self.res_dir}")
        print(f"    scale    {scale_str}")

    def init_models(self):
        self.init_timesteps()
        self.init_scheduler()
        self.init_res_model(train=False, instantiate_only=True)
        self.init_text_models()
        self.init_vae(instantiate_only=True)
        self.init_unet()
        self.init_lora(instantiate_only=True)
        self.init_det_model(train=False, instantiate_only=True)

        ck = torch.load(self.cfg.val.weights, map_location="cpu", weights_only=True)
        self.res_model.load_state_dict(ck["res_model"], strict=True)
        self.unet.load_state_dict(ck["lora_unet"], strict=False)
        self.det_model.load_state_dict(ck["det_model"], strict=True)

    def prepare_all(self):
        attrs = ["res_model", "vae", "text_encoder", "unet", "det_model"]

        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)

    def init_dataset(self):
        self.dataset_format = "coco"
        exts = ["png", "jpg", "jpeg", "JPG", "JPEG"]
        self.img_paths = sorted(sum([glob(os.path.join(self.args.input, f"*.{e}")) for e in exts], []))

    @torch.no_grad()
    def evaluate(self):
        self.det_model.eval()

        pbar = tqdm(self.img_paths)
        for img_path in pbar:
            pbar.set_description(f"Processing {img_path}")

            img = Image.open(img_path).convert("RGB")
            if self.args.scale is None:
                # long side of image is resized to 512
                scale = 512 / max(img.size[0], img.size[1])
                img = img.resize((int(round(x * scale)) for x in img.size), Image.BICUBIC)
            else:
                img = img.resize((int(x * self.args.scale) for x in img.size), Image.BICUBIC)
            img = torch.Tensor((np.array(img) / 255.0).astype(np.float32))
            img = rearrange(img, 'h w c -> c h w').contiguous().float().to(self.device)

            self.lq_list = [img]
            self.lq = self.list2batch(self.lq_list, img_size=max(img.size()))
            self.prompt = [""]

            self.lq = pad_if_smaller(self.lq, size=512)
            self.lq = pad_to_multiples_of(self.lq, multiple=64)

            with self.accelerator.autocast():
                pre_res, z_pre_res, c_txt = self.prepare_condition(batch_size=1)
                z_res = self.partial_diff_and_sample(z_pre_res, z_pre_res, c_txt)
                res = wavelet_reconstruction(self.decode_image(z_res), pre_res)
                res_list = self.batch2list(res, self.lq_list)
                pred, _ = self.det_model(res_list)

            self.save_boxed_images(res_list, pred, [os.path.basename(img_path)])

        pbar.close()

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
