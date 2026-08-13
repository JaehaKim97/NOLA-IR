import os

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from einops import rearrange
from glob import glob
from PIL import Image
from tqdm import tqdm

from pipeline.optical_character_recognition.nola_ir import NOLAIRPipeline
from utils.common import log_txt_as_img, wavelet_reconstruction


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
        self.character = self.cfg.dataset.character
        self.batch_max_length = self.cfg.dataset.batch_max_length
        self.prediction_type = self.cfg.model.ocr_model.params.Prediction.lower()

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
        scale_str = f"{self.args.scale:.2f}" if self.args.scale is not None else "auto (width 512)"
        print(f"NOLA-IR  |  task: {self.args.task}  |  device: {self.device}")
        print(f"    config   {self.args.config}")
        print(f"    weights  {self.cfg.val.weights}")
        print(f"    input    {self.args.input}  ({len(self.img_paths)} images)")
        print(f"    output   {self.res_dir}")
        print(f"    scale    {scale_str}")

    def init_models(self):
        self.init_timesteps()
        self.init_scheduler()
        self.init_converter()
        self.init_res_model(train=False, instantiate_only=True)
        self.init_text_models()
        self.init_vae(instantiate_only=True)
        self.init_unet()
        self.init_lora(instantiate_only=True)
        self.init_ocr_model(train=False, instantiate_only=True)

        ck = torch.load(self.cfg.val.weights, map_location="cpu", weights_only=True)
        self.res_model.load_state_dict(ck["res_model"], strict=True)
        self.unet.load_state_dict(ck["lora_unet"], strict=False)
        self.ocr_model.load_state_dict(ck["ocr_model"], strict=True)

    def prepare_all(self):
        attrs = ["res_model", "vae", "text_encoder", "unet", "ocr_model"]

        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)

    def init_dataset(self):
        exts = ["png", "jpg", "jpeg", "JPG", "JPEG", "PNG"]
        self.img_paths = sorted(sum([glob(os.path.join(self.args.input, f"*.{e}")) for e in exts], []))

    @torch.no_grad()
    def evaluate(self):
        self.ocr_model.eval()

        pbar = tqdm(self.img_paths)
        for img_path in pbar:
            pbar.set_description(f"Processing {img_path}")

            img = Image.open(img_path).convert("RGB")
            if self.args.scale is not None:
                raise NotImplementedError("Manual scale is not supported for OCR demo. Please set --scale to None.")

            w0, h0 = img.size
            img = img.resize((256, 64), Image.BICUBIC)
            img = torch.Tensor((np.array(img) / 255.0).astype(np.float32))
            img = rearrange(img, 'h w c -> c h w').contiguous().float().to(self.device).unsqueeze(0)

            lq_tile = img.repeat(1,1,8,2)
            self.prompt = [""]
            
            with self.accelerator.autocast():
                pre_res_tile, z_pre_res_tile, c_txt = self.prepare_condition(lq=lq_tile, batch_size=lq_tile.size(0))
                z_res_tile = self.partial_diff_and_sample(z_pre_res_tile, z_pre_res_tile, c_txt)
                res_tile = wavelet_reconstruction(self.decode_image(z_res_tile), pre_res_tile)
                res = self.tile2img(res_tile)[0:1]
                pred = self.ocr_model(res, is_train=False)
                pred_label = self.convert2label(pred, batch_size=1)

            res = F.interpolate(res.float(), size=(int(512/w0*h0), 512), mode="bicubic", align_corners=False).clamp(0, 1)
            self.save_batch_images(res, [os.path.basename(img_path)], [""], pred_label, flatten=True, exist_gt=False)

        pbar.close()

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
