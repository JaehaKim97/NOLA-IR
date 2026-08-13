import io
import json
import os
import time
import random

import numpy as np
import torch
from glob import glob
from PIL import Image
from typing import Dict, Mapping, Any, Optional
from torch.utils.data import Dataset

from utils.common import instantiate_from_config


def PairedOCRDataset(**kwargs):
    dataset_type = kwargs.pop("format", "lmdb").lower()
    if dataset_type == "lmdb":
        return PairedLMDBDataset(**kwargs)
    if dataset_type == "img":
        return PairedIMGDataset(**kwargs)
    raise ValueError(f"Unknown dataset_type: {dataset_type} (expected 'lmdb' or 'img')")


class PairedLMDBDataset(Dataset):
    def __init__(
        self,
        root: str,
        file_backend_cfg: Mapping[str, Any],
        gt_size: int,
        data_length: int = -1,
        return_file_name=True,
    ) -> "PairedLMDBDataset":
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.root = root
        self.gt_size = gt_size
        self.data_length = int(data_length)
        self.return_file_name = bool(return_file_name)

        # load meta data for (image, label)
        pairs = []
        with open(os.path.join(root, "data.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
        self.pairs = pairs

    def __len__(self) -> int:
        return self.data_length if self.data_length > 0 else len(self.pairs)

    def load_items(self, gt_path: str, lq_path: str, max_retry: int=5) -> Optional[np.ndarray]:
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

        assert image_gt.shape == image_lq.shape
        assert image_gt.shape[:2] == (self.gt_size[1], self.gt_size[0])

        # hwc, rgb, 0,255, uint8
        return image_gt, image_lq

    def __getitem__(self, index: int, max_retry: int=5) -> Dict[str, torch.Tensor]:
        # load gt, lq images
        img_gt, img_lq = None, None
        while (img_gt is None) or (img_lq is None):
            # load meta file
            filename, label = self.pairs[index]["image"], self.pairs[index]["label"]
            gt_path = os.path.join(self.root, "gt", filename + ".png")
            lq_path = os.path.join(self.root, "lq", filename + ".png")
        
            img_gt, img_lq = self.load_items(gt_path, lq_path)
            prompt = ""
            if (img_gt is None) or (img_lq is None):
                print(f"failed to load {gt_path} and {lq_path}, try another image")
                index = random.randint(0, len(self) - 1)

        # shape: (h, w, c); channel order: RGB; image range: [0, 1], float32.
        img_hq = torch.from_numpy((img_gt / 255.0).transpose(2, 0, 1).astype(np.float32))
        img_lq = torch.from_numpy((img_lq / 255.0).transpose(2, 0, 1).astype(np.float32))

        data = {
            "hq": img_hq,
            "lq": img_lq,
            "label": label,
            "txt": prompt,
        }
        if self.return_file_name:
            data["filename"] = gt_path

        return data


class PairedIMGDataset(Dataset):
    def __init__(
        self,
        root: str,
        file_backend_cfg: Mapping[str, Any],
        gt_size: int,
        data_length: int = -1,
        return_file_name=True,
    ) -> "PairedIMGDataset":
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.root = root
        self.gt_size = gt_size
        self.data_length = int(data_length)
        self.return_file_name = bool(return_file_name)

        self.images = glob(os.path.join(root, "gt", "*.png"))
        
        for idx in range(10):
            self.__getitem__(idx)  # check data loading

    def __len__(self) -> int:
        return self.data_length if self.data_length > 0 else len(self.images)

    def load_items(self, gt_path: str, lq_path: str, max_retry: int=5) -> Optional[np.ndarray]:
        image_gt, image_lq = None, None
        while (image_gt is None) or (image_lq is None):
            try:
                image_gt = Image.open(gt_path).convert('RGB')  # for color image
                image_lq = Image.open(lq_path).convert('RGB')  # for color image
            except IOError:
                print(f'Corrupted image for {gt_path} or {lq_path}')
                raise NotImplementedError("Error occurs in data loading process")

        image_gt = np.array(image_gt.resize(self.gt_size, Image.BICUBIC))
        image_lq = np.array(image_lq.resize(self.gt_size, Image.BICUBIC))

        assert image_gt.shape == image_lq.shape
        assert image_gt.shape[:2] == (self.gt_size[1], self.gt_size[0])
        
        label = os.path.splitext(gt_path)[0].split("/")[-1].split("-")[0]

        # hwc, rgb, 0,255, uint8
        return image_gt, image_lq, label

    def __getitem__(self, index: int, max_retry: int=5) -> Dict[str, torch.Tensor]:
        # load gt, lq images
        img_gt, img_lq, label = None, None, None
        while (img_gt is None) or (img_lq is None):
            # load meta file
            gt_path = self.images[index]
            lq_path = gt_path.replace("gt", "lq")
        
            img_gt, img_lq, label = self.load_items(gt_path, lq_path)
            prompt = ""
            if (img_gt is None) or (img_lq is None):
                print(f"failed to load {gt_path} and {lq_path}, try another image")
                index = random.randint(0, len(self) - 1)

        # shape: (h, w, c); channel order: RGB; image range: [0, 1], float32.
        img_hq = torch.from_numpy((img_gt / 255.0).transpose(2, 0, 1).astype(np.float32))
        img_lq = torch.from_numpy((img_lq / 255.0).transpose(2, 0, 1).astype(np.float32))

        data = {
            "hq": img_hq,
            "lq": img_lq,
            "label": label,
            "txt": prompt,
        }
        if self.return_file_name:
            data["filename"] = gt_path

        return data
