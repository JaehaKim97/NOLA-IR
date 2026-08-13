from typing import Tuple, Set, List, Dict

import torch
from torch import nn

from .clip import FrozenOpenCLIPEmbedder
from .controlnet import ControlledUnetModel, ControlNet
from .distributions import DiagonalGaussianDistribution
from .vae import AutoencoderKL
from .tilevae import VAEHook


def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class ControlLDM(nn.Module):

    def __init__(
        self,
        unet_cfg,
        vae_cfg,
        clip_cfg,
        controlnet_cfg,
        latent_scale_factor,
    ):
        super().__init__()
        self.unet = ControlledUnetModel(**unet_cfg)
        self.vae = AutoencoderKL(**vae_cfg)
        self.clip = FrozenOpenCLIPEmbedder(**clip_cfg)
        self.controlnet = ControlNet(**controlnet_cfg)
        self.scale_factor = latent_scale_factor
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def load_pretrained_sd(self, sd: Dict[str, torch.Tensor], is_turbo: bool=False) -> Set[str]:
        module_map = {
            "unet": "model.diffusion_model",
            "vae": "first_stage_model",
        }
        if is_turbo:
            module_map["clip"] = "conditioner.embedders.0"
        else:
            module_map["clip"] = "cond_stage_model"
        
        modules = [("unet", self.unet), ("vae", self.vae), ("clip", self.clip)]
        used = set()
        for name, module in modules:
            init_sd = {}
            scratch_sd = module.state_dict()
            for key in scratch_sd:
                target_key = ".".join([module_map[name], key])
                init_sd[key] = sd[target_key].clone()
                used.add(target_key)
            module.load_state_dict(init_sd, strict=True)
        unused = set(sd.keys()) - used
        # NOTE: this is slightly different from previous version, which haven't switched
        # the UNet to eval mode and disabled the requires_grad flag.
        frozen_modules = [self.clip, self.unet]
        # NOTE: paramters of self.vae is frozen in the module itself.
        for module in frozen_modules:
            module.eval()
            module.train = disabled_train
            for p in module.parameters():
                p.requires_grad = False
        return unused
    
    @torch.no_grad()
    def load_controlnet_from_ckpt(self, sd: Dict[str, torch.Tensor]) -> None:
        self.controlnet.load_state_dict(sd, strict=True)

    @torch.no_grad()
    def load_controlnet_from_unet(self) -> Tuple[Set[str]]:
        unet_sd = self.unet.state_dict()
        scratch_sd = self.controlnet.state_dict()
        init_sd = {}
        init_with_new_zero = set()
        init_with_scratch = set()
        for key in scratch_sd:
            if key in unet_sd:
                this, target = scratch_sd[key], unet_sd[key]
                if this.size() == target.size():
                    init_sd[key] = target.clone()
                else:
                    d_ic = this.size(1) - target.size(1)
                    oc, _, h, w = this.size()
                    zeros = torch.zeros((oc, d_ic, h, w), dtype=target.dtype)
                    init_sd[key] = torch.cat((target, zeros), dim=1)
                    init_with_new_zero.add(key)
            else:
                init_sd[key] = scratch_sd[key].clone()
                init_with_scratch.add(key)
        self.controlnet.load_state_dict(init_sd, strict=True)
        return init_with_new_zero, init_with_scratch
    
    def vae_encode(
        self,
        image: torch.Tensor,
        sample: bool = True,
        tiled: bool = False,
        tile_size: int = -1,
    ) -> torch.Tensor:
        if tiled:
            def encoder(x: torch.Tensor) -> DiagonalGaussianDistribution:
                h = VAEHook(
                    self.vae.encoder,
                    tile_size=tile_size,
                    is_decoder=False,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(x)
                moments = self.vae.quant_conv(h)
                posterior = DiagonalGaussianDistribution(moments)
                return posterior
        else:
            encoder = self.vae.encode

        if sample:
            z = encoder(image).sample() * self.scale_factor
        else:
            z = encoder(image).mode() * self.scale_factor
        return z
    
    def vae_decode(
        self,
        z: torch.Tensor,
        tiled: bool = False,
        tile_size: int = -1,
    ) -> torch.Tensor:
        if tiled:
            def decoder(z):
                z = self.vae.post_quant_conv(z)
                dec = VAEHook(
                    self.vae.decoder,
                    tile_size=tile_size,
                    is_decoder=True,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(z)
                return dec
        else:
            decoder = self.vae.decode
        return decoder(z / self.scale_factor)
    
    def prepare_condition(self, clean: torch.Tensor, prompt: List[str]) -> Dict[str, torch.Tensor]:
        if prompt is None:
            prompt = [""] * clean.size(0)
        return dict(
            c_txt=self.clip.encode(prompt),
            c_img=self.vae_encode(clean * 2 - 1, sample=False)
        )
    
    def forward(self, x_noisy, t, cond, woSD=False):
        c_txt = cond["c_txt"]
        c_img = cond["c_img"]
        control = self.controlnet(
            x=x_noisy, hint=c_img,
            timesteps=t, context=c_txt
        )
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        eps = self.unet(
            x=x_noisy, timesteps=t,
            context=c_txt, control=control, only_mid_control=False
        )
        return eps
