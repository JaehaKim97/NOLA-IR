import io
import os
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CocoDetection, VOCSegmentation

from utils.common import instantiate_from_config
from utils.segmentation import (
    Compose, ConvertCocoPolysToMask, FilterAndRemapCocoCategories, _coco_remove_images_without_annotations
)


def PairedSEGDataset(**kwargs):
    dataset_type = kwargs.pop("format", "voc").lower()
    if dataset_type == "voc":
        return PairedVOCDataset(**kwargs)
    if dataset_type == "coco":
        return PairedCOCODataset(**kwargs)
    raise ValueError(f"Unknown dataset_type: {dataset_type} (expected 'coco' or 'voc')")


class PairedVOCDataset(VOCSegmentation):
    def __init__(
        self,
        root: str,
        pair_subdir: str,
        file_backend_cfg: Mapping[str, Any],
        gt_size: int,
        data_length: int = -1,
        return_file_name=True,
        # voc related setting
        n_classes: int = 21,
        year: str = '2012',
        image_set: str = 'val',
        download: bool = False,
        transform: transforms = None,
        target_transform: transforms = None,
    ) -> "PairedVOCDataset":
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.pair_subdir = str(pair_subdir)
        self.gt_size = int(gt_size)
        self.data_length = int(data_length)
        self.return_file_name = bool(return_file_name)

        # voc dataset initialization
        self.n_classes = n_classes
        super(PairedVOCDataset, self).__init__(root, year, image_set, download, transform, target_transform)
    
    def __len__(self) -> int:
        return self.data_length if self.data_length > 0 else len(self.images)

    def load_items(self, gt_path: str, lq_path: str, mask_path: str, max_retry: int=5) -> Optional[np.ndarray]:
        gt_bytes, lq_bytes = None, None
        while (gt_bytes is None) or (lq_bytes is None):
            if max_retry == 0:
                return None
            gt_bytes = self.file_backend.get(gt_path)
            lq_bytes = self.file_backend.get(lq_path)
            max_retry -= 1
            if (gt_bytes is None) or (lq_bytes is None):
                time.sleep(0.5)
        image_gt = Image.open(io.BytesIO(gt_bytes)).convert("RGB")
        image_lq = Image.open(io.BytesIO(lq_bytes)).convert("RGB")

        assert (image_gt.size == image_lq.size) and (min(image_gt.size) == self.gt_size)

        mask = Image.open(mask_path)
        mask = mask.resize((image_gt.size[0], image_gt.size[1]), Image.NEAREST)
        image_gt, image_lq, mask = np.array(image_gt), np.array(image_lq), np.array(mask)

        # hwc, rgb, 0,255, uint8
        return image_gt, image_lq, mask

    def __getitem__(self, index: int, max_retry: int=5) -> Dict[str, torch.Tensor]:
        # load gt, lq images
        img_gt, img_lq = None, None
        while (img_gt is None) or (img_lq is None):
            # load meta file
            img_path, mask_path = self.images[index], self.masks[index]
            gt_path = img_path.replace("JPEGImages", os.path.join(self.pair_subdir, "gt")).replace('.jpg', '.png')
            lq_path = img_path.replace("JPEGImages", os.path.join(self.pair_subdir, "lq")).replace('.jpg', '.png')
            img_gt, img_lq, mask = self.load_items(gt_path, lq_path, mask_path)
            prompt = ""
            if (img_gt is None) or (img_lq is None):
                print(f"failed to load {gt_path}")
                raise NotImplementedError

        # shape: (h, w, c); channel order: RGB; image range: [0, 1], float32.
        img_hq = torch.from_numpy((img_gt / 255.0).transpose(2, 0, 1).astype(np.float32))
        img_lq = torch.from_numpy((img_lq / 255.0).transpose(2, 0, 1).astype(np.float32))
        mask = torch.from_numpy(mask).long()

        data = {
            "hq": img_hq,
            "lq": img_lq,
            "mask": mask,
            "txt": prompt,
        }
        if self.return_file_name:
            data["filename"] = gt_path

        return data


class PairedCOCODataset(CocoDetection):
    def __init__(
        self,
        root: str,
        pair_subdir: str,
        file_backend_cfg: Mapping[str, Any],
        gt_size: int,
        data_length: int = -1,
        return_file_name=True,
        # voc related setting
        n_classes: int = 21,
        image_set: str = 'val',
        exclude_no_annotation: bool = True,
    ) -> "PairedCOCODataset":
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.gt_size = int(gt_size)
        self.data_length = int(data_length)
        self.return_file_name = bool(return_file_name)

        # coco dataset initialization
        self.n_classes = n_classes
        self.gt_folder = os.path.join(root, pair_subdir, "gt")
        self.lq_folder = os.path.join(root, pair_subdir, "lq")
        ann_file = os.path.join(root, f"annotations/instances_{image_set}2017.json")
        # use only labels from voc categories
        CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
        self.transform_coco_det2seg = Compose([FilterAndRemapCocoCategories(CAT_LIST, remap=True), ConvertCocoPolysToMask()])
        super(PairedCOCODataset, self).__init__(self.gt_folder, ann_file)
        if exclude_no_annotation:
            _coco_remove_images_without_annotations(self, CAT_LIST)
    
    def __len__(self) -> int:
        return self.data_length if self.data_length > 0 else len(self.ids)

    def load_items(self, id: int, max_retry: int=5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        gt_bytes, lq_bytes = None, None
        while (gt_bytes is None) or (lq_bytes is None):
            if max_retry == 0:
                return None
            gt_path = os.path.join(self.root, self.coco.loadImgs(id)[0]["file_name"])
            gt_path = os.path.splitext(gt_path)[0] + ".png"
            gt_bytes = self.file_backend.get(gt_path)
            lq_path = gt_path.replace(self.gt_folder, self.lq_folder)
            lq_bytes = self.file_backend.get(lq_path)
            max_retry -= 1
            if (gt_bytes is None) or (lq_bytes is None):
                time.sleep(0.5)
        image_gt = Image.open(io.BytesIO(gt_bytes)).convert("RGB")
        image_lq = Image.open(io.BytesIO(lq_bytes)).convert("RGB")

        assert (image_gt.size == image_lq.size) and (min(image_gt.size) == self.gt_size)

        annot = self.coco.loadAnns(self.coco.getAnnIds(id))
        img_info = self.coco.loadImgs(annot[0]["image_id"])[0]
        height, width = img_info["height"], img_info["width"]
        _, mask = self.transform_coco_det2seg(Image.new("RGB", (width, height)), annot)
        mask = mask.resize((image_gt.size[0], image_gt.size[1]), Image.NEAREST)
        image_gt, image_lq, mask = np.array(image_gt), np.array(image_lq), np.array(mask)

        # hwc, rgb, 0,255, uint8
        return image_gt, image_lq, mask, gt_path

    def __getitem__(self, index: int, max_retry: int=5) -> Dict[str, torch.Tensor]:
        id = self.ids[index]
        # load gt, lq images
        img_gt, img_lq = None, None
        while (img_gt is None) or (img_lq is None):
            # load meta file
            img_gt, img_lq, mask, gt_path = self.load_items(id)
            prompt = ""
            if (img_gt is None) or (img_lq is None):
                print(f"failed to load {gt_path}")
                raise NotImplementedError

        # shape: (h, w, c); channel order: RGB; image range: [0, 1], float32.
        img_hq = torch.from_numpy((img_gt / 255.0).transpose(2, 0, 1).astype(np.float32))
        img_lq = torch.from_numpy((img_lq / 255.0).transpose(2, 0, 1).astype(np.float32))
        mask = torch.from_numpy(mask).long()

        data = {
            "hq": img_hq,
            "lq": img_lq,
            "mask": mask,
            "txt": prompt,
        }
        if self.return_file_name:
            data["filename"] = gt_path

        return data
