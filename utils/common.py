import importlib
import logging
import os
import sys
import time
import warnings
from copy import deepcopy
from contextlib import contextmanager
from shutil import copyfile
from typing import Any, Callable, Literal, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torch.nn import functional as F
from torchvision.models import get_model
from torchvision.transforms.functional import normalize
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")


def get_obj_from_str(string: str, reload: bool=False) -> Any:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Mapping[str, Any]) -> Any:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def copy_opt_file(opt_file, experiments_root):
    # copy the yml file to the experiment root
    cmd = 'accelerate launch ' + ' '.join(sys.argv)
    filename = os.path.join(experiments_root, opt_file)
    os.makedirs(os.path.join(experiments_root, os.path.dirname(opt_file)), exist_ok=True)
    copyfile(opt_file, filename)

    with open(filename, 'r+') as f:
        lines = f.readlines()
        lines.insert(0, f'# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n')
        f.seek(0)
        f.writelines(lines)


def set_logger(file_name, exp_dir, logger_name):
    logger = logging.getLogger(file_name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(f'{exp_dir}/{logger_name}')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    streamHandler.setLevel(level=logging.INFO)  # Save and print
    fileHandler.setLevel(level=logging.DEBUG)  # Only save
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    return logger


class Logger:
    def __init__(self, name, path, accelerator, logger_name="logger.log"):
        self.logger = set_logger(name, path, logger_name)
        self.accelerator = accelerator
        
    def __call__(self, text, print=True):
        if self.accelerator.is_local_main_process:
            if print:
                self.logger.info(text)
            else:
                self.logger.debug(text)


def wavelet_blur(image: Tensor, radius: int):
    """
    Apply wavelet blur to the input tensor.
    """
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    # apply convolution
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output


def wavelet_decomposition(image: Tensor, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq

    return high_freq, low_freq


def wavelet_reconstruction(content_feat:Tensor, style_feat:Tensor):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    # calculate the wavelet decomposition of the content feature
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    # calculate the wavelet decomposition of the style feature
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    # reconstruct the content feature with the style's high frequency
    return content_high_freq + style_low_freq


def log_txt_as_img(wh, xc, font_path="assets/NanumGothic.ttf", font_size=18):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        try:
            font = ImageFont.truetype(font_path, size=font_size)
        except OSError:
            print(f"[WARN] Cannot load font at {font_path}, using default font instead.")
            # font = ImageFont.truetype('font/DejaVuSans.ttf', size=size)
            font = ImageFont.load_default()
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return (txts + 1) / 2


# https://github.com/XPixelGroup/BasicSR/blob/033cd6896d898fdd3dcda32e3102a792efa1b8f4/basicsr/utils/color_util.py#L186
def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB images to YCbCr images (PyTorch version).

    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    Args:
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
         y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img


# https://github.com/XPixelGroup/BasicSR/blob/033cd6896d898fdd3dcda32e3102a792efa1b8f4/basicsr/metrics/psnr_ssim.py#L52
def calculate_psnr_pt(img, img2, crop_border=0, test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2)**2, dim=[1, 2, 3])
    return 10. * torch.log10(1. / (mse + 1e-8))


def calculate_psnr_list(
    img_list: list[Tensor],
    img2_list: list[Tensor],
    crop_border: int = 0,
    test_y_channel: bool = False,
) -> list[float]:
    """Calculate PSNR for pairs of CHW tensors provided as lists."""

    if len(img_list) != len(img2_list):
        raise ValueError("img_list and img2_list must have the same length")
    if len(img_list) == 0:
        return []

    out = []
    for img, img2 in zip(img_list, img2_list):
        if img.shape != img2.shape:
            raise ValueError(f"Image shapes differ: {img.shape} vs {img2.shape}")
        # add batch dim so we can reuse calculate_psnr_pt
        psnr = calculate_psnr_pt(img.unsqueeze(0), img2.unsqueeze(0), crop_border, test_y_channel)
        out.append(float(psnr.item()))
    return out


def calculate_psnr_pt_masked(img, img2, binary_mask, crop_border=0, test_y_channel=False):
    """Masked PSNR. binary_mask: (n, h, w) with 0/1 elements."""
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
        binary_mask = binary_mask[:, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mask = binary_mask.to(torch.float64).unsqueeze(1)   # (n, h, w) -> (n, 1, h, w)

    se = (img - img2) ** 2 * mask                       # zero out non-masked pixels
    n_channels = img.shape[1]
    valid = mask.sum(dim=[1, 2, 3]) * n_channels        # masked-pixel count * channels, per image

    mse = se.sum(dim=[1, 2, 3]) / (valid + 1e-8)
    return 10. * torch.log10(1. / (mse + 1e-8))


def calculate_psnr_list_masked(
    img_list: list[Tensor],
    img2_list: list[Tensor],
    mask_list: list[Tensor],
    crop_border: int = 0,
    test_y_channel: bool = False,
) -> list[float]:
    """Calculate PSNR for pairs of CHW tensors provided as lists."""

    if len(img_list) != len(img2_list):
        raise ValueError("img_list and img2_list must have the same length")
    if len(img_list) != len(mask_list):
        raise ValueError("img_list and mask_list must have the same length")
    if len(img_list) == 0:
        return []

    out = []
    for img, img2, mask in zip(img_list, img2_list, mask_list):
        if img.shape != img2.shape:
            raise ValueError(f"Image shapes differ: {img.shape} vs {img2.shape}")
        # add batch dim so we can reuse calculate_psnr_pt
        psnr = calculate_psnr_pt_masked(img.unsqueeze(0), img2.unsqueeze(0), mask.unsqueeze(0), crop_border, test_y_channel)
        out.append(float(psnr.item()))
    return out


def calculate_lpips_pt(img, img2, net_lpips, crop_border=8, img_range=1.0, **kwargs):
    """Computes the PSNR (Peak-Signal-Noise-Ratio) in batch"""
        
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    
    # norm to [-1, 1]
    img = normalize(img, mean, std)
    img2 = normalize(img2, mean, std)

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
        
    lpips = net_lpips(img, img2).squeeze(1,2,3)  # batch-wise lpips
    return lpips


def _print_different_keys_loading(crt_net, load_net, strict=True):
    """Print keys with different name or different size when loading models.

    1. Print keys with different names.
    2. If strict=False, print the same key but with different tensor size.
        It also ignore these keys with different sizes (not load).

    Args:
        crt_net (torch model): Current network.
        load_net (dict): Loaded network.
        strict (bool): Whether strictly loaded. Default: True.
    """
    crt_net = crt_net.state_dict()
    crt_net_keys = set(crt_net.keys())
    load_net_keys = set(load_net.keys())

    if crt_net_keys != load_net_keys:
        print('Current net - loaded net:')
        for v in sorted(list(crt_net_keys - load_net_keys)):
            print(f'  {v}')
        print('Loaded net - current net:')
        for v in sorted(list(load_net_keys - crt_net_keys)):
            print(f'  {v}')

    # check the size for the same keys
    if not strict:
        common_keys = crt_net_keys & load_net_keys
        for k in common_keys:
            if crt_net[k].size() != load_net[k].size():
                print(f'Size different, ignore [{k}]: crt_net: '
                      f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                load_net[k + '.ignore'] = load_net.pop(k)


def load_network(crt_net, load_path, strict):
    """Load network.

    Args:
        load_path (str): The path of networks to be loaded.
        net (nn.Module): Network.
        strict (bool): Whether strictly loaded.
        param_key (str): The parameter key of loaded network. If set to
            None, use the root 'path'.
            Default: 'params'.
    """
    if os.path.exists(load_path):
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        _print_different_keys_loading(crt_net, load_net, strict)
        crt_net.load_state_dict(load_net, strict=strict)
    else:
        try:
            load_net = get_model('ResNet18', weights=load_path, num_classes=1000,).state_dict()
            _print_different_keys_loading(crt_net, load_net, strict)
            crt_net.load_state_dict(load_net, strict=strict)
        except:
            raise NotImplementedError(f'{load_path} is not valid model path!')
        
    return crt_net


def pad_if_smaller(imgs: torch.Tensor, size: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    ph, pw = max(size - h, 0), max(size - w, 0)
    return F.pad(imgs, pad=(0, pw, 0, ph), mode="constant", value=0)


def pad_to_multiples_of(imgs: torch.Tensor, multiple: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h % multiple == 0 and w % multiple == 0:
        return imgs.clone()
    ph, pw = map(lambda x: (x + multiple - 1) // multiple * multiple - x, (h, w))
    return F.pad(imgs, pad=(0, pw, 0, ph), mode="constant", value=0)


def sliding_windows(h: int, w: int, tile_size: int, tile_stride: int) -> Tuple[int, int, int, int]:
    hi_list = list(range(0, h - tile_size + 1, tile_stride))
    if (h - tile_size) % tile_stride != 0:
        hi_list.append(h - tile_size)
    
    wi_list = list(range(0, w - tile_size + 1, tile_stride))
    if (w - tile_size) % tile_stride != 0:
        wi_list.append(w - tile_size)
    
    coords = []
    for hi in hi_list:
        for wi in wi_list:
            coords.append((hi, hi + tile_size, wi, wi + tile_size))
    return coords


# https://github.com/csslc/CCSR/blob/main/model/q_sampler.py#L503
def gaussian_weights(tile_width: int, tile_height: int) -> np.ndarray:
    """Generates a gaussian mask of weights for tile contributions"""
    latent_width = tile_width
    latent_height = tile_height
    var = 0.01
    midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
    x_probs = [
        np.exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / np.sqrt(2 * np.pi * var)
        for x in range(latent_width)]
    midpoint = latent_height / 2
    y_probs = [
        np.exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / np.sqrt(2 * np.pi * var)
        for y in range(latent_height)]
    weights = np.outer(y_probs, x_probs)
    return weights


def make_tiled_fn(
    fn: Callable[[torch.Tensor], torch.Tensor],
    size: int,
    stride: int,
    scale_type: Literal["up", "down"] = "up",
    scale: int = 1,
    channel: int | None = None,
    weight: Literal["uniform", "gaussian"] = "gaussian",
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    # callback: Callable[[int, int, int, int], None] | None = None,
    progress: bool = True,
) -> Callable[[torch.Tensor], torch.Tensor]:
    # Only split the first input of function.
    def tiled_fn(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if scale_type == "up":
            scale_fn = lambda n: int(n * scale)
        else:
            scale_fn = lambda n: int(n // scale)

        b, c, h, w = x.size()
        out_dtype = dtype or x.dtype
        out_device = device or x.device
        out_channel = channel or c
        out = torch.zeros(
            (b, out_channel, scale_fn(h), scale_fn(w)),
            dtype=out_dtype,
            device=out_device,
        )
        count = torch.zeros_like(out, dtype=torch.float32)
        weight_size = scale_fn(size)
        weights = (
            gaussian_weights(weight_size, weight_size)[None, None]
            if weight == "gaussian"
            else np.ones((1, 1, weight_size, weight_size))
        )
        weights = torch.tensor(
            weights,
            dtype=out_dtype,
            device=out_device,
        )

        indices = sliding_windows(h, w, size, stride)
        pbar = tqdm(
            indices, desc=f"Tiled Processing", disable=not progress, leave=False
        )
        for hi, hi_end, wi, wi_end in pbar:
            x_tile = x[..., hi:hi_end, wi:wi_end]
            out_hi, out_hi_end, out_wi, out_wi_end = map(
                scale_fn, (hi, hi_end, wi, wi_end)
            )
            if len(args) or len(kwargs):
                kwargs.update(dict(hi=hi, hi_end=hi_end, wi=wi, wi_end=wi_end))
            out[..., out_hi:out_hi_end, out_wi:out_wi_end] += (
                fn(x_tile, *args, **kwargs) * weights
            )
            count[..., out_hi:out_hi_end, out_wi:out_wi_end] += weights
        out = out / count
        return out

    return tiled_fn


def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class VAEForwardWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x=None, z=None, mode="encode"):
        if mode == "encode":
            posterior = self.vae.encode(x).latent_dist
            return posterior
        elif mode == "decode":
            return self.vae.decode(z).sample
        else:
            raise ValueError(mode)


def replace_controlnet(
    controlnet,
    in_channels=4,
    out_channels=320,
):
    """
    Replace controlnet.controlnet_cond_embedding with a latent-space embedding:
    (B,4,H/8,W/8) -> (B,320,H/8,W/8)

    Keeps device/dtype consistent with the original module.
    """

    old_emb = controlnet.controlnet_cond_embedding
    device = next(old_emb.parameters()).device
    dtype  = next(old_emb.parameters()).dtype

    new_emb = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    ).to(device=device, dtype=dtype)

    for m in new_emb.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0.0, mode="fan_in", nonlinearity="linear")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    controlnet.controlnet_cond_embedding = new_emb

    return controlnet


@contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout


class CombinedLoader:
    def __init__(self, loaders: dict):
        self.loaders = loaders
        self._iters = {name: iter(loader) for name, loader in self.loaders.items()}

    def __iter__(self):
        return self

    def _next_from_loader(self, name):
        try:
            batch = next(self._iters[name])
        except StopIteration:
            self._iters[name] = iter(self.loaders[name])
            batch = next(self._iters[name])
        return batch

    def __next__(self):
        return {name: self._next_from_loader(name) for name in self.loaders.keys()}
