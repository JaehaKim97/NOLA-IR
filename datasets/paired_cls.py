import io
import os
import time
import random

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from typing import Dict, Mapping, Any, Optional

from utils.common import instantiate_from_config


class PairedCLSDataset(ImageFolder):
    def __init__(
        self,
        root: str,
        file_backend_cfg: Mapping[str, Any],
        data_length=-1,
        return_file_name=True,
    ) -> "PairedCLSDataset":
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.root = root
        self.gt_root = os.path.join(root, 'gt')
        self.lq_root = os.path.join(root, 'lq')
        self.data_length = int(data_length)
        self.return_file_name = bool(return_file_name)

        # initialize the ImageFolder
        super(PairedCLSDataset, self).__init__(os.path.join(root, 'gt'))

    def __len__(self) -> int:
        return self.data_length if self.data_length > 0 else len(self.imgs)

    def load_paired_images(self, gt_path: str, lq_path: str, max_retry: int=5) -> Optional[np.ndarray]:
        gt_bytes, lq_bytes = None, None
        while (gt_bytes is None) or (lq_bytes is None):
            if max_retry == 0:
                return None
            gt_bytes = self.file_backend.get(gt_path)
            lq_bytes = self.file_backend.get(lq_path)
            max_retry -= 1
            if (gt_bytes is None) or (lq_bytes is None):
                time.sleep(0.5)

        image_gt = np.array(Image.open(io.BytesIO(gt_bytes)).convert("RGB"))
        image_lq = np.array(Image.open(io.BytesIO(lq_bytes)).convert("RGB"))

        # hwc, rgb, 0,255, uint8
        return image_gt, image_lq

    def __getitem__(self, index: int, max_retry: int=5) -> Dict[str, torch.Tensor]:
        # load gt, lq images
        img_gt, img_lq = None, None
        while (img_gt is None) or (img_lq is None):
            # load meta file
            gt_path, label = self.imgs[index]
            lq_path = gt_path.replace(self.gt_root, self.lq_root)
            img_gt, img_lq = self.load_paired_images(gt_path, lq_path)
            prompt = ""
            if (img_gt is None) or (img_lq is None):
                print(f"failed to load {gt_path}")
                raise NotImplementedError

        # shape: (c, h, w); channel order: RGB; image range: [0, 1], float32.
        img_gt = torch.from_numpy((img_gt / 255.0).transpose(2, 0, 1).copy()).float()
        img_lq = torch.from_numpy((img_lq / 255.0).transpose(2, 0, 1).copy()).float()

        data = {
            "hq": img_gt,
            "lq": img_lq,
            "txt": prompt,
            "label": label,
        }
        if self.return_file_name:
            data["filename"] = gt_path

        return data
