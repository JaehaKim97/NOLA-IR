import io
import os
import xml.etree.ElementTree as ET
import time
import torch
import random
import numpy as np

from PIL import Image
from typing import Dict, Mapping, Any, Optional
from torchvision import transforms
from torchvision.datasets import CocoDetection, VOCDetection

from utils.common import instantiate_from_config
from datasets.utils import get_label2id, convert2coco


def PairedDETDataset(**kwargs):
    dataset_type = kwargs.pop("format", "voc").lower()
    if dataset_type == "voc":
        return PairedVOCDataset(**kwargs)
    if dataset_type == "coco":
        return PairedCOCODataset(**kwargs)
    raise ValueError(f"Unknown dataset_type: {dataset_type} (expected 'coco' or 'voc')")


class PairedVOCDataset(VOCDetection):
    def __init__(
        self,
        root: str,
        pair_subdir: str,
        file_backend_cfg: Mapping[str, Any],
        gt_size: int,
        data_length: int = -1,
        return_file_name=True,
        # voc related setting
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
        super(PairedVOCDataset, self).__init__(root, year, image_set, download, transform, target_transform)
        self.label2id = get_label2id('./datasets/labels/voc.txt')

    def __len__(self) -> int:
        return self.data_length if self.data_length > 0 else len(self.images)

    def load_items(self, gt_path: str, lq_path: str, annot_path: str, max_retry: int=5) -> Optional[np.ndarray]:
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
        annot = self.parse_voc_xml(ET.parse(annot_path).getroot())

        height, width = int(annot['annotation']['size']['height']), int(annot['annotation']['size']['width'])
        if height >= width:
            scale_factor = self.gt_size / height
            height, width = self.gt_size, int(width * scale_factor)
        else:
            scale_factor = self.gt_size / width
            height, width = int(height * scale_factor), self.gt_size

        assert (image_gt.shape[:2] == (height, width))

        for item in annot['annotation']['object']:
            xmin, xmax = item['bndbox']["xmin"], item['bndbox']["xmax"]
            ymin, ymax = item['bndbox']["ymin"], item['bndbox']["ymax"]
            item['bndbox']["xmin"] = str(max(int(int(xmin) * scale_factor), 1))
            item['bndbox']["xmax"] = str(min(int(int(xmax) * scale_factor), width))
            item['bndbox']["ymin"] = str(max(int(int(ymin) * scale_factor), 1))
            item['bndbox']["ymax"] = str(min(int(int(ymax) * scale_factor), height))

        # convert to coco style
        annot = convert2coco(annot, self.label2id)

        # hwc, rgb, 0,255, uint8
        return image_gt, image_lq, annot

    def __getitem__(self, index: int, max_retry: int=5) -> Dict[str, torch.Tensor]:
        # load gt, lq images
        img_gt, img_lq = None, None
        while (img_gt is None) or (img_lq is None):
            # load meta file
            img_path, annot_path = self.images[index], self.annotations[index]
            gt_path = img_path.replace("JPEGImages", os.path.join(self.pair_subdir, "gt")).replace('.jpg', '.png')
            lq_path = img_path.replace("JPEGImages", os.path.join(self.pair_subdir, "lq")).replace('.jpg', '.png')
            img_gt, img_lq, annot = self.load_items(gt_path, lq_path, annot_path)
            prompt = ""
            if (img_gt is None) or (img_lq is None):
                print(f"failed to load {gt_path}")
                raise NotImplementedError

        # shape: (h, w, c); channel order: RGB; image range: [0, 1], float32.
        img_hq = torch.from_numpy((img_gt / 255.0).transpose(2, 0, 1).astype(np.float32))
        img_lq = torch.from_numpy((img_lq / 255.0).transpose(2, 0, 1).astype(np.float32))
        annot = {k: torch.Tensor(v).long() if isinstance(v, list) else v for k, v in annot.items()}

        data = {
            "hq": img_hq,
            "lq": img_lq,
            "annot": annot,
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
        # coco related setting
        image_set: str = 'val',
        transform: transforms = None,
        target_transform: transforms = None,
        exclude_no_annotation: bool = True,
    ) -> "PairedCOCODataset":
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.gt_size = int(gt_size)
        self.data_length = int(data_length)
        self.return_file_name = bool(return_file_name)

        # coco dataset initialization
        self.gt_folder = os.path.join(root, pair_subdir, "gt")
        self.lq_folder = os.path.join(root, pair_subdir, "lq")
        ann_file = os.path.join(root, f"annotations/instances_{image_set}2017.json")
        super(PairedCOCODataset, self).__init__(self.gt_folder, ann_file, transform, target_transform)
        if exclude_no_annotation:
            # exclude samples without annotations; number of no-annotations samples: 48
            self.ids = [id for id in self.ids if len(self.coco.loadAnns(self.coco.getAnnIds(id))) > 0]

    def __len__(self) -> int:
        return self.data_length if self.data_length > 0 else len(self.ids)

    def load_items(self, id: int, max_retry: int=5) -> Optional[np.ndarray]:
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
        image_gt = np.array(Image.open(io.BytesIO(gt_bytes)).convert("RGB"))
        image_lq = np.array(Image.open(io.BytesIO(lq_bytes)).convert("RGB"))
        
        annot = self.coco.loadAnns(self.coco.getAnnIds(id))
        annot = [obj for obj in annot if obj["iscrowd"] == 0]
        # convert annotation format
        # list to dict, (x1, y1, x2, y2) format
        if len(annot) > 0:
            img_info = self.coco.loadImgs(annot[0]["image_id"])[0]
            height, width = img_info["height"], img_info["width"]
            
            boxes = [obj["bbox"] for obj in annot]
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(min=0, max=width)
            boxes[:, 1::2].clamp_(min=0, max=height)
            classes = [obj["category_id"] for obj in annot]
            classes = torch.tensor(classes, dtype=torch.int64)
            area = torch.tensor([obj["area"] for obj in annot])
            iscrowd = torch.tensor([obj["iscrowd"] for obj in annot])
            
            # image resizing
            if height >= width:
                scale_factor = self.gt_size / height
                height, width = self.gt_size, int(width * scale_factor)
            else:
                scale_factor = self.gt_size / width
                height, width = int(height * scale_factor), self.gt_size
            
            assert (image_gt.shape[:2] == (height, width))
            
            xmin, xmax = boxes[:,0].clone(), boxes[:,2].clone()
            ymin, ymax = boxes[:,1].clone(), boxes[:,3].clone()
            boxes[:,0] = torch.maximum(xmin * scale_factor, torch.tensor(1.0))
            boxes[:,2] = torch.minimum(xmax * scale_factor, torch.tensor(width))
            boxes[:,1] = torch.maximum(ymin * scale_factor, torch.tensor(1.0))
            boxes[:,3] = torch.minimum(ymax * scale_factor, torch.tensor(height))
            
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
        return image_gt, image_lq, annot, gt_path
        
    def __getitem__(self, index: int) -> tuple[Any, Any]:
        id = self.ids[index]
        # loag gt image
        img_gt, img_lq = None, None
        while (img_gt is None) or (img_lq is None):
            # load meta file
            img_gt, img_lq, annot, gt_path = self.load_items(id)
            prompt = ""
            if (img_gt is None) or (img_lq is None):
                print(f"failed to load {gt_path}")
                raise NotImplementedError

        # shape: (h, w, c); channel order: RGB; image range: [0, 1], float32.
        img_hq = torch.from_numpy((img_gt / 255.0).transpose(2, 0, 1).astype(np.float32))
        img_lq = torch.from_numpy((img_lq / 255.0).transpose(2, 0, 1).astype(np.float32))
        annot = {k: torch.Tensor(v).long() if isinstance(v, list) else v for k, v in annot.items()}

        data = {
            "hq": img_hq,
            "lq": img_lq,
            "annot": annot,
            "txt": prompt,
        }
        if self.return_file_name:
            data["filename"] = gt_path

        return data
