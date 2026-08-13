import io
import random
import time
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from datasets.augment import augment
from datasets.realesrgan import RealESRGANDegradationMixin
from datasets.utils import center_crop_arr, random_crop_arr
from utils.common import instantiate_from_config


class RealesrganCLSDataset(RealESRGANDegradationMixin, ImageFolder):
    def __init__(
        self,
        root: str,
        file_backend_cfg: Mapping[str, Any],
        gt_size: int,
        out_size: int,
        crop_type: str,
        hflip: bool,
        rotation: bool,
        # blur kernel settings of the first degradation stage
        blur_kernel_size,
        kernel_list,
        kernel_prob,
        blur_sigma,
        betag_range,
        betap_range,
        sinc_prob,
        # blur kernel settings of the second degradation stage
        blur_kernel_size2,
        kernel_list2,
        kernel_prob2,
        blur_sigma2,
        betag_range2,
        betap_range2,
        sinc_prob2,
        final_sinc_prob,
        data_length=-1,
        return_file_name=False,
    ) -> "RealesrganCLSDataset":
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.gt_size = int(gt_size)
        self.out_size = int(out_size)
        self.crop_type = str(crop_type)
        assert self.crop_type in ["none", "center", "random"]
        self.hflip = bool(hflip)
        self.rotation = bool(rotation)
        self.data_length = int(data_length)
        self.return_file_name = bool(return_file_name)

        # real-esrgan degradation
        self.init_degradation(
            blur_kernel_size, kernel_list, kernel_prob, blur_sigma, betag_range, betap_range, sinc_prob,
            blur_kernel_size2, kernel_list2, kernel_prob2, blur_sigma2, betag_range2, betap_range2, sinc_prob2,
            final_sinc_prob,
        )

        # initialize the ImageFolder
        super(RealesrganCLSDataset, self).__init__(root)

    def __len__(self) -> int:
        return self.data_length if self.data_length > 0 else len(self.imgs)

    def load_gt_image(self, image_path: str, max_retry: int = 5) -> Optional[np.ndarray]:
        image_bytes = None
        tries = 0
        while image_bytes is None:
            if tries >= max_retry:
                return None
            image_bytes = self.file_backend.get(image_path)
            if image_bytes is None:
                time.sleep(0.2)
            tries += 1

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        w, h = image.size
        if w >= h:
            image = image.resize((int(self.gt_size * w / h), self.gt_size), Image.BICUBIC)
        else:
            image = image.resize((self.gt_size, int(self.gt_size * h / w)), Image.BICUBIC)

        if self.crop_type == "none":
            if image.height != self.out_size and image.width != self.out_size:
                raise ValueError(f"crop_type='none', {(image.width, image.height)} vs out_size={self.out_size}")
            image = np.array(image)
        elif self.crop_type == "center":
            image = center_crop_arr(image, self.out_size)
        elif self.crop_type == "random":
            image = random_crop_arr(image, self.out_size)
        else:
            raise ValueError(self.crop_type)

        image = augment(image, self.hflip, self.rotation)
        return image

    def __getitem__(self, index: int, max_retry: int = 5) -> Dict[str, torch.Tensor]:
        index = index % len(self.imgs)

        # load gt image
        img_gt = None
        while img_gt is None:
            gt_path, label = self.imgs[index]
            img_gt = self.load_gt_image(gt_path)
            prompt = ""
            if img_gt is None:
                print(f"filed to load {gt_path}, try another image")
                index = random.randint(0, len(self) - 1)

       # shape: (c, h, w); channel order: RGB; image range: [0, 1], float32.
        img_hq = torch.from_numpy((img_gt / 255.0).astype(np.float32).transpose(2, 0, 1).copy()).float()
        kernel, kernel2, sinc_kernel = self.sample_degradation_kernels()

        data = {
            "hq": img_hq,
            "kernel1": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
            "txt": prompt,
            "label": label,
        }
        if self.return_file_name:
            data["filename"] = gt_path

        return data
