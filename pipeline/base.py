import math
import os
import warnings
from copy import deepcopy
from typing import Dict, List

import diffusers
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from utils.common import (
    Logger, copy_opt_file, disabled_train, instantiate_from_config,
    load_network, replace_controlnet, VAEForwardWrapper
)
from utils.sampler import SpacedSampler
from utils.tabulate import tabulate


class CorePipeline:

    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.init_environment()
        self.summary_models()

    def init_environment(self):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if self.args.seed is not None:
            self.cfg.seed = self.args.seed
        set_seed(self.cfg.seed)

        self.exp_dir = self.cfg.exp_dir
        subdirs = {
            "ckpt_dir": "checkpoints",
            "img_dir": "images",
            "res_dir": "results",
        }
        for attr, name in subdirs.items():
            setattr(self, attr, os.path.join(self.exp_dir, name))

        accelerator = Accelerator(
            dataloader_config=DataLoaderConfiguration(split_batches=True),
            mixed_precision=self.cfg.precision,
        )

        os.makedirs(self.exp_dir, exist_ok=True)
        for attr in subdirs.keys():
            os.makedirs(getattr(self, attr), exist_ok=True)

        self.log = Logger(__name__, self.exp_dir, accelerator, logger_name="logger.log")
        self.log(f"Using seed: {self.cfg.seed}")
        self.log(f"Active pipeline: {self.cfg.pipeline}")
        self.log(f"Experiment folder created at: {self.exp_dir}")
        copy_opt_file(self.args.config, self.exp_dir)
        self.model_to_be_save = []
        self.submodule_to_be_save = {}

        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_warning()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        mp = accelerator.mixed_precision
        weight_dtype = torch.float16 if mp == "fp16" else torch.bfloat16 if mp == "bf16" else torch.float32
        # self.log(f"Using {weight_dtype} for model weights")

        self.accelerator = accelerator
        self.weight_dtype = weight_dtype
        self.device = accelerator.device

    def get_model_load_path(self, model_name: str):
        load_path = None
        strict = True
        
        if self.args.eval_only:
            if self.cfg.val.get("resume"):
                load_path = getattr(self.cfg.val.resume, model_name, None)
            else:
                candidate_path = os.path.join(self.ckpt_dir, f"{model_name}_last.pt")
                if os.path.exists(candidate_path):
                    load_path = candidate_path
        else:
            if self.cfg.train.get("resume"):
                load_path = getattr(self.cfg.train.resume, model_name, None)
                strict = bool(self.cfg.train.resume.get("strict_load", False))
        
        return load_path, strict

    def init_timesteps(self):
        self.t_start = self.cfg.model.diffusion.start
        self.timesteps = [
            math.floor(self.t_start / self.cfg.model.diffusion.num_timesteps * i)
            for i in range(1, self.cfg.model.diffusion.num_timesteps + 1)
        ]
        self.timesteps_pt = torch.as_tensor(self.timesteps, device=self.device, dtype=torch.int64)
        # self.log(f"Timesteps: {self.timesteps}, total number of {len(self.timesteps)}")

    def init_scheduler(self):
        self.scheduler = DDPMScheduler.from_pretrained(self.cfg.model.sd_path, subfolder="scheduler")
        self.spaced_sampler = SpacedSampler(self.scheduler.betas)
        self.spaced_sampler.make_schedule(len(self.timesteps), used_timesteps=self.timesteps)
        self.spaced_sampler.to(self.device)

    def init_res_model(self, train=True, instantiate_only=False):
        self.res_model = instantiate_from_config(self.cfg.model.res_model)
        if instantiate_only: return self.res_model.eval().requires_grad_(False)

        load_path, strict = self.get_model_load_path("res_model")
        if load_path:
            self.res_model = load_network(self.res_model, load_path, strict=strict)
            self.log(f"RES model: Load from: {load_path}")
        else:
            self.log("RES model: Initialize from SCRATCH")

        if train:
            self.res_model.train().requires_grad_(True)
        else:
            self.res_model.eval().requires_grad_(False)
        self.model_to_be_save.append("res_model")

    def init_text_models(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.cfg.model.sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.model.sd_path,
            subfolder="text_encoder",
            torch_dtype=self.weight_dtype
        )
        self.text_encoder.eval().requires_grad_(False)

    def init_vae(self, instantiate_only=False):
        self.vae = AutoencoderKL.from_pretrained(
            self.cfg.model.sd_path,
            subfolder="vae",
            # torch_dtype=self.weight_dtype,
        )
        self.vae.eval().requires_grad_(False)
        self.train_vae = False
        if instantiate_only: return

        load_path, strict = self.get_model_load_path("vae_decoder")
        if load_path:
            self.vae.decoder = load_network(self.vae.decoder, load_path, strict=strict)
            self.log(f"VAE Decoder: Load from: {load_path}")

        self.train_vae = (not self.args.eval_only) and self.cfg.model.get("train_vae", False)
        if self.train_vae:
            warnings.filterwarnings("ignore", message=r"Grad strides do not match*", category=UserWarning)
            self.scaling_factor = self.vae.config.scaling_factor
            # disable training mode for encoder and conv layers
            self.vae.encoder.train = disabled_train
            self.vae.quant_conv.train = disabled_train
            self.vae.post_quant_conv.train = disabled_train
            self.vae.decoder.train().requires_grad_(True)
            self.vae = VAEForwardWrapper(self.vae)
            self.submodule_to_be_save["vae_decoder"] = ("vae", "vae.decoder")
            self.log("VAE Decoder: Set to be TRAINABLE")

    def init_unet(self):
        self.unet = UNet2DConditionModel.from_pretrained(
            self.cfg.model.sd_path,
            subfolder="unet",
            torch_dtype=self.weight_dtype
        )
        self.unet.eval().requires_grad_(False)

    def init_lora(self, instantiate_only=False):
        lora_cfg = LoraConfig(
            r=self.cfg.model.lora.rank,
            lora_alpha=self.cfg.model.lora.rank,
            init_lora_weights="gaussian",
            target_modules=self.cfg.model.lora.target_modules,
        )
        self.unet.add_adapter(lora_cfg)
        if instantiate_only: return

        load_path, strict = self.get_model_load_path("lora_unet")
        if load_path:
            m, u = self.unet.load_state_dict(torch.load(load_path), strict=False)
            if u: raise RuntimeError(f"Unexpected keys ({len(u)}): {u[:5]} ...")
            self.log(f"LoRA-UNet: Load from: {load_path}")

        if not self.args.eval_only:
            lora_params = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
            assert lora_params, "Failed to find lora parameters"
            for p in lora_params:
                p.data = p.to(torch.float32)
            self.lora_params = lora_params
            self.lora_to_be_save = {"lora_unet": "unet"}

        if self.cfg.val.get("fuse_lora", False):
            self.log("Fusing LoRA into UNet for evaluation")
            self.unet.fuse_lora(lora_scale=1.0, safe_fusing=True)
            self.unet.unload_lora()

    def init_controlnet(self, train=True):
        self.controlnet = ControlNetModel.from_unet(self.unet)
        self.controlnet = replace_controlnet(self.controlnet)

        load_path, strict = self.get_model_load_path("controlnet")
        if load_path:
            self.controlnet = load_network(self.controlnet, load_path, strict=strict)
            self.log(f"ControlNet: Load from: {load_path}")

        if train:
            self.controlnet.train().requires_grad_(True)
        else:
            self.controlnet.eval().requires_grad_(False)
        self.model_to_be_save.append("controlnet")

    def init_cldm(self, train=True):
        self.cldm = instantiate_from_config(self.cfg.model.cldm)
        sd = torch.load(self.cfg.model.sd_path + "/v2-1_512-ema-pruned.ckpt",
                        map_location="cpu",
                        weights_only=False
        )["state_dict"]
        self.cldm.load_pretrained_sd(sd)
        # we don't use clip and vae in ControlLDM
        delattr(self.cldm, "clip")
        delattr(self.cldm, "vae")
        load_path, strict = self.get_model_load_path("controlnet")
        if load_path:
            self.cldm.load_controlnet_from_ckpt(torch.load(load_path, map_location="cpu"))
            self.log(f"ControlNet: Load from: {load_path}")
        else:
            self.cldm.load_controlnet_from_unet()
            self.log("ControlNet: Initialize from UNet")
        self.cldm.eval().requires_grad_(False)
        if train:
            self.cldm.controlnet.train().requires_grad_(True)
        self.submodule_to_be_save["controlnet"] = ("cldm", "controlnet")

    def init_disc(self):
        warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated as an API.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=r"Grad strides do not match*", category=UserWarning)
        self.disc = instantiate_from_config(self.cfg.model.disc)
        load_path, strict = self.get_model_load_path("disc")
        if load_path:
            self.disc = load_network(self.disc, load_path, strict=strict)
            self.log(f"DISC model: Load from: {load_path}")
        else:
            self.log("DISC model: Initialize from SCRATCH")

        self.disc.train().requires_grad_(True)
        self.model_to_be_save.append("disc")

    def summary_models(self):
        table_data = []
        for attr, value in self.__dict__.items():
            if not isinstance(value, torch.nn.Module):
                continue
            model = value
            model_type = type(model).__name__
            total_params = sum(p.numel() for p in model.parameters()) / 1_000_000
            learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
            table_data.append([attr, model_type, f"{total_params:.2f}", f"{learnable_params:.2f}"])
        headers = ["Model Name", "Model Type", "Total Parameters (M)", "Learnable Parameters (M)"]
        table = tabulate(table_data, headers=headers, tablefmt="pretty")
        self.log(f"Model Summary:\n{table}")

    def on_training_start(self):
        self.global_step = 0
        self.max_steps = self.cfg.train.train_steps
        self.log(f"Training for {self.max_steps:05d} steps")
        self.epoch = 0
        self.loss_avg_records = deepcopy(self.loss_records)
        if self.accelerator.is_local_main_process:
            self.writer = SummaryWriter(self.exp_dir)

    def make_pbar(self, total: int, desc: str = None, leave: bool = True) -> tqdm:
        """Create a progress bar with consistent styling."""
        return tqdm(
            iterable=None,
            disable=not self.accelerator.is_local_main_process,
            unit="batch",
            total=total,
            desc=desc,
            leave=leave,
        )

    def encode_prompt(self, prompt: List[str]) -> Dict[str, torch.Tensor]:
        txt_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        # text_encoder2 = TextEncoderForwardWrapper(self.text_encoder, txt_ids.to(self.accelerator.device))
        # self.get_complexity(text_encoder2, input_size=(77,), print_per_layer_stat=True)
        text_embed = self.text_encoder(txt_ids.to(self.accelerator.device))[0]
        return text_embed

    def encode_image(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.train_vae:
            img_embed = self.vae(x=image*2-1, mode="encode").mode() * self.scaling_factor
        else:
            img_embed = self.vae.encode(image*2-1).latent_dist.mode() * self.vae.config.scaling_factor
        return img_embed

    def decode_image(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.train_vae:
            img_decoded = (self.vae(z=z/self.scaling_factor, mode="decode") + 1) / 2
        else:
            img_decoded = (self.vae.decode(z/self.vae.config.scaling_factor).sample.float() + 1) / 2
        return img_decoded

    @torch.no_grad()
    def prepare_condition(self, lq=None, gt=None, prompt=None, return_gt=False, batch_size=None):
        if lq is None:
            lq = self.lq
        if prompt is None:
            prompt = self.prompt
        if batch_size is None:
            batch_size = self.gt.size(0)

        pre_res = self.res_model(lq[:batch_size])
        if return_gt:
            if gt is None:
                gt = self.gt
            z_gt = self.encode_image(gt[:batch_size])
        z_pre_res = self.encode_image(pre_res)
        c_txt = self.encode_prompt(prompt[:batch_size])

        if return_gt:
            return pre_res, z_pre_res, c_txt, z_gt
        return pre_res, z_pre_res, c_txt

    def t2index(self, t):
        return (t.unsqueeze(1) == torch.tensor(self.timesteps).to(t).unsqueeze(0)).int().argmax(dim=1)

    def masked_l1(self, a, b, margin):
        if margin > 0:
            diff = F.l1_loss(a, b, reduction="none")
            return torch.where(diff > margin, diff, diff.new_zeros(())).mean()
        else:
            return F.l1_loss(a, b)

    def calculate_tfd(self, feat_x, feat_y, target_layers=None):
        if target_layers is None:
            target_layers = self.get_target_layers()

        feat_x_avg = sum(feat_x[k] for k in target_layers) / len(target_layers)
        feat_y_avg = sum(feat_y[k] for k in target_layers) / len(target_layers)
        tfd = F.l1_loss(feat_x_avg, feat_y_avg, reduction='none').mean(dim=(1, 2, 3))
        return tfd

    def log_training_metrics(self):
        for key, values in self.loss_records.items():
            t = torch.tensor(values, device=self.device, dtype=torch.float32)
            self.loss_avg_records[key] = self.accelerator.gather_for_metrics(t).mean().item()
            values.clear()

        if not self.accelerator.is_local_main_process:
            return

        loss_str = ", ".join(f"{k}: {v:.4f}" for k, v in self.loss_avg_records.items())
        self.log(
            f"[{self.global_step:05d}/{self.max_steps:05d}] Training loss: ( {loss_str} )",
            print=False,
        )

        for key, avg in self.loss_avg_records.items():
            self.writer.add_scalar(f"loss/{key}", avg, self.global_step)

        for name, obj in self.__dict__.items():
            if not name.startswith("opt_"):
                continue
            if not isinstance(obj, torch.optim.Optimizer):
                continue
            if not obj.param_groups:
                continue
            self.writer.add_scalar(
                f"train/learning_rate_{name.removeprefix('opt_')}",
                obj.param_groups[0]["lr"],
                self.global_step,
            )

    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        return model
    
    def save_checkpoints(self, last=False):
        if self.accelerator.is_local_main_process:
            postfix = "last" if last else f"{self.global_step:07d}"
            for model_name in self.model_to_be_save:
                model = self.unwrap_model(getattr(self, model_name))
                torch.save(model.state_dict(), f"{self.ckpt_dir}/{model_name}_{postfix}.pt")

            submodule_to_be_save = getattr(self, "submodule_to_be_save", {}) or {}
            for savename, (model_attr, sub_attrs) in submodule_to_be_save.items():
                submodule = self.unwrap_model(getattr(self, model_attr))
                for sub_attr in sub_attrs.split("."):
                    submodule = getattr(submodule, sub_attr, None)
                torch.save(submodule.state_dict(), f"{self.ckpt_dir}/{savename}_{postfix}.pt")

            lora_to_be_save = getattr(self, "lora_to_be_save", {}) or {}
            for savename, model_attr in lora_to_be_save.items():
                model = self.unwrap_model(getattr(self, model_attr))
                lora_state = {
                    name: param.detach().cpu()
                    for name, param in model.named_parameters()
                    if "lora_" in name
                }
                assert len(lora_state) > 0, "No LoRA parameters found to save"
                torch.save(lora_state, f"{self.ckpt_dir}/{savename}_{postfix}.pt")

    def get_complexity(self, model, input_size=(3, 512, 512), print_per_layer_stat=False, verbose=True):
        try:
            from ptflops import get_model_complexity_info
            macs, params = get_model_complexity_info(
                model,
                input_size,
                as_strings=True,
                backend='pytorch',
                print_per_layer_stat=print_per_layer_stat,
                verbose=verbose
            )
            print(f"Model Complexity for {type(model).__name__}:")
            print(f"FLOPs: {macs}, Parameters: {params}")
            return macs, params
        except ImportError:
            self.log("ptflops is not installed. Install it with `pip install ptflops` to get model complexity.")
            return None, None


class TextEncoderForwardWrapper(torch.nn.Module):
    def __init__(self, text_encoder, txt_ids):
        super().__init__()
        self.text_encoder = text_encoder
        self.txt_ids = txt_ids

    def forward(self, x):
        text_embed = self.text_encoder(self.txt_ids)[0]
        return text_embed
