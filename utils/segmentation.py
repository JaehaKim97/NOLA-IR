import copy

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask


@torch.inference_mode()
def mask2rgb(mask):
    template = torch.zeros_like(mask).expand(3, *mask.shape).to(torch.float64)
    max_rgb = 255.0

    # color maps for 21 classes:
    color_list = [
        [158, 184, 217],    # background
        [124, 147, 195],    # aeroplane
        [162, 87, 114],     # bicycle
        [97, 163, 186],     # bird
        [255, 255, 221],    # boat
        [210, 222, 50],     # bottle
        [162, 197, 121],    # bus
        [162, 197, 121],    # car
        [0, 66, 90],        # cat
        [31, 138, 112],     # chair
        [191, 219, 56],     # cow
        [252, 115, 0],      # diningtable
        [131, 162, 255],    # dog
        [180, 189, 255],    # horse
        [255, 227, 187],    # motorbike
        [255, 210, 143],    # person
        [251, 236, 178],    # pottedplant
        [248, 189, 235],    # sheep
        [82, 114, 242],     # sofa
        [7, 37, 65],        # train
        [188, 122, 249],    # tvmonitor
    ]
    # color_list = [
    #     [0,   0,   0],      # background
    #     [128, 0,   0],      # aeroplane
    #     [0,   128, 0],      # bicycle
    #     [128, 128, 0],      # bird
    #     [0,   0,   128],    # boat
    #     [128, 0,   128],    # bottle
    #     [0,   128, 128],    # bus
    #     [128, 128, 128],    # car
    #     [64,  0,   0],      # cat
    #     [192, 0,   0],      # chair
    #     [64,  128, 0],      # cow
    #     [192, 128, 0],      # diningtable
    #     [64,  0,   128],    # dog
    #     [192, 0,   128],    # horse
    #     [64,  128, 128],    # motorbike
    #     [192, 128, 128],    # person
    #     [0,   64,  0],      # pottedplant
    #     [128, 64,  0],      # sheep
    #     [0,   192, 0],      # sofa
    #     [128, 192, 0],      # train
    #     [0,   64,  128],    # tvmonitor
    # ]

    dontcare = (mask==max_rgb)
    if dontcare.sum() != 0:
        color = [0, 0, 0]  # color_list[0]
        color = np.array(color) / max_rgb
        template[:, dontcare] = torch.Tensor(color).to(template.dtype).to(template.device).view(3,1).repeat(1, dontcare.sum())

    for idx in range(21):
        mask_idx = (mask==idx)
        if mask_idx.sum() != 0:
            color = color_list[idx]
            color = np.array(color) / max_rgb
            template[:, mask_idx] = torch.Tensor(color).to(template.dtype).to(template.device).view(3,1).repeat(1,mask_idx.sum())

    template = template.permute(1,0,2,3)

    return template


def calculate_mat(pred, target, n):
    k = (pred >= 0) & (pred < n)
    inds = n * pred[k].to(torch.int64) + target[k]
    return torch.bincount(inds, minlength=n**2).reshape(n, n)


def compute_iou(mat):
    h = mat.float()
    iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
    return iu


class FilterAndRemapCocoCategories:
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image, anno):
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            return image, anno
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        return image, anno


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask:
    def __call__(self, image, anno):
        w, h = image.size
        segmentations = [obj["segmentation"] for obj in anno]
        cats = [obj["category_id"] for obj in anno]
        if segmentations:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
            cats = torch.as_tensor(cats, dtype=masks.dtype)
            # merge all instance masks into a single segmentation map
            # with its corresponding categories
            target, _ = (masks * cats[:, None, None]).max(dim=0)
            # discard overlapping instances
            target[masks.sum(0) > 1] = 255
        else:
            target = torch.zeros((h, w), dtype=torch.uint8)
        target = Image.fromarray(target.numpy())
        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None, min_area: float = 1000):
    def _has_valid_annotation(anno):
        if len(anno) == 0:
            return False
        return sum(obj.get("area", 0) for obj in anno) > min_area

    valid_img_ids = []
    for img_id in dataset.ids:
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list is not None:
            anno = [obj for obj in anno if obj.get("category_id") in cat_list]
        if _has_valid_annotation(anno):
            valid_img_ids.append(img_id)

    # mutate in-place so it can be called from __init__
    dataset.ids = valid_img_ids
    return dataset


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
