import io
import os
import random
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CocoDetection, VOCSegmentation

from datasets.augment import augment
from datasets.realesrgan import RealESRGANDegradationMixin
from datasets.utils import center_crop_arr, random_crop_arr
from utils.common import instantiate_from_config
from utils.segmentation import (
    Compose, ConvertCocoPolysToMask, FilterAndRemapCocoCategories, _coco_remove_images_without_annotations
)


def RealesrganSEGDataset(**kwargs):
    dataset_type = kwargs.pop("format", "voc").lower()
    if dataset_type == "voc":
        return RealesrganVOCDataset(**kwargs)
    if dataset_type == "coco":
        return RealesrganCOCODataset(**kwargs)
    raise ValueError(f"Unknown dataset_type: {dataset_type} (expected 'coco' or 'voc')")


class RealesrganVOCDataset(RealESRGANDegradationMixin, VOCSegmentation):
    def __init__(
        self,
        root: str,
        file_backend_cfg: Mapping[str, Any],
        gt_size: int,
        resize_range: Sequence[int],
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
        data_length: int = -1,
        return_file_name: bool = False,
        # voc related settings
        year: str = '2012',
        image_set: str = 'train',
        download: bool = False,
        transform: transforms = None,
        target_transform: transforms = None,
    ) -> "RealesrganVOCDataset":
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.gt_size = int(gt_size)
        self.resize_range = resize_range
        self.out_size = out_size
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

        # voc dataset initialization
        super(RealesrganVOCDataset, self).__init__(root, year, image_set, download, transform, target_transform)

    def __len__(self) -> int:
        return self.data_length if self.data_length > 0 else len(self.images)

    def load_items(self, image_path: str, mask_path: str, max_retry: int=5) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask = Image.open(mask_path)

        if self.resize_range is not None:
            r = random.uniform(*self.resize_range)
        else:
            r = 1.0
        w, h = image.size
        if w >= h:
            image = image.resize((int(self.gt_size * w / h * r), int(self.gt_size * r)), Image.BICUBIC)
            mask = mask.resize((int(self.gt_size * w / h * r), int(self.gt_size * r)), Image.NEAREST)
        else:
            image = image.resize((int(self.gt_size * r), int(self.gt_size * h / w * r)), Image.BICUBIC)
            mask = mask.resize((int(self.gt_size * r), int(self.gt_size * h / w * r)), Image.NEAREST)

        image, mask = np.array(image), np.array(mask)
        min_size = min(mask.shape)
        if (self.out_size is not None) and min_size < self.out_size:
            ow, oh = mask.shape
            padh = self.out_size - oh if oh < self.out_size else 0
            padw = self.out_size - ow if ow < self.out_size else 0
            image = np.pad(image, ((0, padw), (0, padh), (0,0)), mode='constant', constant_values=0)
            mask = np.pad(mask, ((0, padw), (0, padh)), mode='constant', constant_values=255)

        if self.crop_type != "none":
            if image.shape[0] == self.out_size and image.shape[1] == self.out_size:
                image, mask = np.array(image), np.array(mask)
            else:
                if self.crop_type == "center":
                    image = center_crop_arr(image, self.out_size)
                    mask = center_crop_arr(mask, self.out_size)
                elif self.crop_type == "random":
                    image, crop_pos = random_crop_arr(image, self.out_size, return_params=True)
                    mask = random_crop_arr(mask, self.out_size, crop_pos=crop_pos)

        image, mask = augment([image, mask], self.hflip, self.rotation)

        # hwc, rgb, 0,255, uint8
        return image, mask

    def __getitem__(self, index: int, max_retry: int=5) -> Dict[str, torch.Tensor]:
        index = index % len(self.images)

        # load gt image
        img_gt = None
        while img_gt is None:
            # load meta file
            gt_path, mask_path = self.images[index], self.masks[index]
            img_gt, mask = self.load_items(gt_path, mask_path)
            prompt = ""
            if img_gt is None:
                print(f"filed to load {gt_path}, try another image")
                index = random.randint(0, len(self) - 1)

        # shape: (c, h, w); channel order: RGB; image range: [0, 1], float32.
        img_hq = torch.from_numpy((img_gt / 255.0).astype(np.float32).transpose(2, 0, 1).copy()).float()
        kernel, kernel2, sinc_kernel = self.sample_degradation_kernels()

        mask = torch.from_numpy(mask).long()
        data = {
            "hq": img_hq,
            "kernel1": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
            "txt": prompt,
            "mask": mask,
        }
        if self.return_file_name:
            data["filename"] = gt_path

        return data


class RealesrganCOCODataset(RealESRGANDegradationMixin, CocoDetection):
    def __init__(
        self,
        root: str,
        file_backend_cfg: Mapping[str, Any],
        gt_size: int,
        resize_range: Sequence[int],
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
        data_length: int = -1,
        return_file_name: bool = False,
        # coco related setting
        image_set: str = 'train',
        exclude_no_annotation: bool = True,
    ) -> "RealesrganCOCODataset":
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.gt_size = int(gt_size)
        self.resize_range = resize_range
        self.out_size = out_size
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

        # coco dataset initialization
        anno_file_template = "{}_{}2017.json"
        PATHS = {
            "train": ("train2017", os.path.join("annotations", anno_file_template.format("instances", "train"))),
            "val": ("val2017", os.path.join("annotations", anno_file_template.format("instances", "val"))),
        }
        img_folder, ann_file = PATHS[image_set]
        img_folder = os.path.join(root, img_folder)
        ann_file = os.path.join(root, ann_file)
        # use only labels from voc categories
        CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
        self.transform_coco_det2seg = Compose([FilterAndRemapCocoCategories(CAT_LIST, remap=True), ConvertCocoPolysToMask()])
        super(RealesrganCOCODataset, self).__init__(img_folder, ann_file)
        if exclude_no_annotation:
            _coco_remove_images_without_annotations(self, CAT_LIST)

    def __len__(self) -> int:
        return self.data_length if self.data_length > 0 else len(self.ids)

    def load_items(self, id: int, max_retry: int=5) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_path = os.path.join(self.root, self.coco.loadImgs(id)[0]["file_name"])
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        annot = self.coco.loadAnns(self.coco.getAnnIds(id))
        image, mask = self.transform_coco_det2seg(image, annot)
        
        if self.resize_range is not None:
            r = random.uniform(*self.resize_range)
        else:
            r = 1.0
        w, h = image.size
        if w >= h:
            image = image.resize((int(self.gt_size * w / h * r), int(self.gt_size * r)), Image.BICUBIC)
            mask = mask.resize((int(self.gt_size * w / h * r), int(self.gt_size * r)), Image.NEAREST)
        else:
            image = image.resize((int(self.gt_size * r), int(self.gt_size * h / w * r)), Image.BICUBIC)
            mask = mask.resize((int(self.gt_size * r), int(self.gt_size * h / w * r)), Image.NEAREST)

        image, mask = np.array(image), np.array(mask)
        min_size = min(mask.shape)
        if (self.out_size is not None) and min_size < self.out_size:
            ow, oh = mask.shape
            padh = self.out_size - oh if oh < self.out_size else 0
            padw = self.out_size - ow if ow < self.out_size else 0
            image = np.pad(image, ((0, padw), (0, padh), (0,0)), mode='constant', constant_values=0)
            mask = np.pad(mask, ((0, padw), (0, padh)), mode='constant', constant_values=255)

        if self.crop_type != "none":
            if image.shape[0] == self.out_size and image.shape[1] == self.out_size:
                image, mask = np.array(image), np.array(mask)
            else:
                if self.crop_type == "center":
                    image = center_crop_arr(image, self.out_size)
                    mask = center_crop_arr(mask, self.out_size)
                elif self.crop_type == "random":
                    image, crop_pos = random_crop_arr(image, self.out_size, return_params=True)
                    mask = random_crop_arr(mask, self.out_size, crop_pos=crop_pos)

        image, mask = augment([image, mask], self.hflip, self.rotation)

        # hwc, rgb, 0,255, uint8
        return image, mask, image_path

    def __getitem__(self, index: int, max_retry: int=5) -> Dict[str, torch.Tensor]:
        id = self.ids[index]
        # load gt image
        img_gt = None
        while img_gt is None:
            # load meta file
            img_gt, mask, gt_path = self.load_items(id)
            prompt = ""
            if img_gt is None:
                index = random.randint(0, len(self) - 1)
                id = self.ids[index]

        # shape: (c, h, w); channel order: RGB; image range: [0, 1], float32.
        img_hq = torch.from_numpy((img_gt / 255.0).astype(np.float32).transpose(2, 0, 1).copy()).float()
        kernel, kernel2, sinc_kernel = self.sample_degradation_kernels()

        mask = torch.from_numpy(mask).long()
        data = {
            "hq": img_hq,
            "kernel1": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
            "txt": prompt,
            "mask": mask,
        }
        if self.return_file_name:
            data["filename"] = gt_path

        return data
