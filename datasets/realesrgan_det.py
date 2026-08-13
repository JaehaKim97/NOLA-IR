import io
import os
import random
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, Mapping, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CocoDetection, VOCDetection

from datasets.realesrgan import RealESRGANDegradationMixin
from datasets.utils import center_crop_arr, convert2coco, get_label2id, random_crop_arr
from utils.common import instantiate_from_config


def RealesrganDETDataset(**kwargs):
    dataset_type = kwargs.pop("format", "voc").lower()
    if dataset_type == "voc":
        return RealesrganVOCDataset(**kwargs)
    if dataset_type == "coco":
        return RealesrganCOCODataset(**kwargs)
    raise ValueError(f"Unknown dataset_type: {dataset_type} (expected 'coco' or 'voc')")


class RealesrganVOCDataset(RealESRGANDegradationMixin, VOCDetection):
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

        # voc dataset initialization
        super(RealesrganVOCDataset, self).__init__(root, year, image_set, download, transform, target_transform)
        self.image_set = image_set
        self.label2id = get_label2id('./datasets/labels/voc.txt')

    def __len__(self) -> int:
        return self.data_length if self.data_length > 0 else len(self.images)

    def load_items(self, image_path: str, annot_path: str, max_retry: int=5) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        annot = self.parse_voc_xml(ET.parse(annot_path).getroot())
        height, width = image.shape[:2]

        # flip augmentation
        if self.hflip and (torch.rand(1) < 0.5):
            image = cv2.flip(image, 1, image)
            for item in annot['annotation']['object']:
                xmin, xmax = int(item['bndbox']["xmin"]), int(item['bndbox']["xmax"])
                item['bndbox']["xmin"] = str(max(width - xmax, 1))  # gurantee minimum value of 1
                item['bndbox']["xmax"] = str(width - xmin)

        # image resizing
        if height >= width:
            scale_factor = self.gt_size / height
            image = cv2.resize(image, dsize=(int(width * scale_factor), self.gt_size), interpolation=cv2.INTER_CUBIC)
        else:
            scale_factor = self.gt_size / width
            image = cv2.resize(image, dsize=(self.gt_size, int(height * scale_factor)), interpolation=cv2.INTER_CUBIC)
        height, width = image.shape[:2]
        for item in annot['annotation']['object']:
            xmin, xmax = int(item['bndbox']["xmin"]), int(item['bndbox']["xmax"])
            ymin, ymax = int(item['bndbox']["ymin"]), int(item['bndbox']["ymax"])
            item['bndbox']["xmin"] = str(max(int(xmin * scale_factor), 1))
            item['bndbox']["xmax"] = str(min(int(xmax * scale_factor), width))
            item['bndbox']["ymin"] = str(max(int(ymin * scale_factor), 1))
            item['bndbox']["ymax"] = str(min(int(ymax * scale_factor), height))

        # image cropping
        if self.crop_type != "none":
            if height == self.out_size and width == self.out_size:
                pass
            else:
                if self.crop_type == "center":
                    image, crop_pos = center_crop_arr(image, self.out_size, return_params=True)
                elif self.crop_type == "random":
                    image, crop_pos = random_crop_arr(image, self.out_size, return_params=True)
                new_y0, new_x0 = crop_pos
                new_obj = []
                for item in annot['annotation']['object']:
                    xmin, xmax = int(item['bndbox']["xmin"]), int(item['bndbox']["xmax"])
                    ymin, ymax = int(item['bndbox']["ymin"]), int(item['bndbox']["ymax"])
                    if (xmax > new_x0) and (ymax > new_y0):
                        xmin, xmax = max(xmin - new_x0, 1), min(xmax - new_x0, self.out_size)
                        ymin, ymax = max(ymin - new_y0, 1), min(ymax - new_y0, self.out_size)
                        threshold = 15
                        if (xmax > xmin + threshold) and (ymax > ymin + threshold): 
                            item['bndbox']["xmin"], item['bndbox']["xmax"] = str(xmin), str(xmax)
                            item['bndbox']["ymin"], item['bndbox']["ymax"] = str(ymin), str(ymax)
                            new_obj.append(item.copy())
                annot['annotation']['object'] = new_obj

        # convert to coco style
        annot = convert2coco(annot, self.label2id)

        # hwc, rgb, 0,255, uint8
        return image, annot

    def __getitem__(self, index: int, max_retry: int=5) -> Dict[str, torch.Tensor]:
        index = index % len(self.images)

        # load gt image
        img_gt, annot_length = None, 0
        while (img_gt is None) or (annot_length == 0):
            # load meta file
            gt_path, annot_path = self.images[index], self.annotations[index]
            img_gt, annot = self.load_items(gt_path, annot_path)
            prompt = ""
            annot_length = len(annot['boxes'])
            if (img_gt is None) or (annot_length == 0):
                print(f"failed to load {gt_path}, try another image")
                index = random.randint(0, len(self) - 1)

        # shape: (c, h, w); channel order: RGB; image range: [0, 1], float32.
        img_hq = torch.from_numpy((img_gt / 255.0).astype(np.float32).transpose(2, 0, 1).copy()).float()
        kernel, kernel2, sinc_kernel = self.sample_degradation_kernels()

        annot = {k: torch.Tensor(v).long() if isinstance(v, list) else v for k, v in annot.items()}
        data = {
            "hq": img_hq,
            "kernel1": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
            "txt": prompt,
            "annot": annot,
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
        transform: transforms = None,
        target_transform: transforms = None,
        exclude_no_annotation: bool = True,
    ) -> "RealesrganCOCODataset":
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.gt_size = int(gt_size)
        self.out_size = int(out_size)        
        self.crop_type = crop_type
        assert self.crop_type in ["none"]
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
        super(RealesrganCOCODataset, self).__init__(img_folder, ann_file, transform, target_transform)
        self.image_set = image_set
        if exclude_no_annotation:
            # exclude samples without annotations; number of no-annotations samples: 1021
            self.ids = [id for id in self.ids if len(self.coco.loadAnns(self.coco.getAnnIds(id))) > 0]

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
        image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        height, width = image.shape[:2]

        annot = self.coco.loadAnns(self.coco.getAnnIds(id))
        annot = [obj for obj in annot if obj["iscrowd"] == 0]
        # convert annotation format
        # list to dict, (x1, y1, x2, y2) format
        if len(annot) > 0:
            boxes = [obj["bbox"] for obj in annot]
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(min=0, max=width)
            boxes[:, 1::2].clamp_(min=0, max=height)
            classes = [obj["category_id"] for obj in annot]
            classes = torch.tensor(classes, dtype=torch.int64)
            area = torch.tensor([obj["area"] for obj in annot])
            iscrowd = torch.tensor([obj["iscrowd"] for obj in annot])

            # flip augmentation
            if self.hflip and (torch.rand(1) < 0.5):
                image = cv2.flip(image, 1, image)
                xmin, xmax = boxes[:,0].clone(), boxes[:,2].clone()
                boxes[:,0] = torch.maximum(width - xmax, torch.tensor(1.0))  # gurantee minimum value of 1
                boxes[:,2] = width - xmin

            # image resizing
            if height >= width:
                scale_factor = self.gt_size / height
                image = cv2.resize(image, dsize=(int(width * scale_factor), self.gt_size), interpolation=cv2.INTER_CUBIC)
            else:
                scale_factor = self.gt_size / width
                image = cv2.resize(image, dsize=(self.gt_size, int(height * scale_factor)), interpolation=cv2.INTER_CUBIC)
            height, width = image.shape[:2]
            xmin, xmax = boxes[:,0].clone(), boxes[:,2].clone()
            ymin, ymax = boxes[:,1].clone(), boxes[:,3].clone()
            boxes[:,0] = torch.maximum(xmin * scale_factor, torch.tensor(1.0))
            boxes[:,2] = torch.minimum(xmax * scale_factor, torch.tensor(width))
            boxes[:,1] = torch.maximum(ymin * scale_factor, torch.tensor(1.0))
            boxes[:,3] = torch.minimum(ymax * scale_factor, torch.tensor(height))

            # image cropping is not supported
            if self.crop_type != "none":
                pass

            # keep only valid labels
            keep = (boxes[:, 3] > boxes[:, 1] + 1) & (boxes[:, 2] > boxes[:, 0] + 1)
            new_annot = {}
            new_annot["image_id"] = annot[0]["image_id"]
            new_annot["boxes"] = boxes[keep]
            new_annot["labels"] = classes[keep]
            new_annot["area"] = area[keep]
            new_annot["iscrowd"] = iscrowd[keep]
            annot = new_annot

        # hwc, rgb, 0,255, uint8
        return image, annot, image_path

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        # -------------------------------- Load hq images -------------------------------- #
        id = self.ids[index]
        # loag gt image
        img_gt, annot_length = None, 0
        while (img_gt is None) or (self.image_set == "train" and annot_length == 0):
            # load meta file
            img_gt, annot, gt_path = self.load_items(id)
            prompt = ""
            annot_length = len(annot)
            if (img_gt is None) or (annot_length == 0):
                index = random.randint(0, len(self) - 1)
                id = self.ids[index]

        # shape: (c, h, w); channel order: RGB; image range: [0, 1], float32.
        img_hq = torch.from_numpy((img_gt / 255.0).astype(np.float32).transpose(2, 0, 1).copy()).float()
        kernel, kernel2, sinc_kernel = self.sample_degradation_kernels()

        annot = {k: torch.Tensor(v).long() if isinstance(v, list) else v for k, v in annot.items()}
        data = {
            "hq": img_hq,
            "kernel1": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
            "txt": prompt,
            "annot": annot,
        }
        if self.return_file_name:
            data["filename"] = gt_path

        return data
