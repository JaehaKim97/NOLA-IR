import bisect
import copy
import io
import math
from collections import defaultdict
from contextlib import redirect_stdout
from itertools import chain, repeat
from typing import Any, Dict, List

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.distributed as dist
import torchvision
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data.sampler import BatchSampler, Sampler
from tqdm import tqdm


def convert2label(label, is_coco=False):
    if is_coco:
        # COCO labels
        table = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', '-', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', '-', 'backpack', 'umbrella', '-', '-',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', '-', 'wine glass', 'cup', 'fork', 'knife',
            'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
            'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
            'bed', '-', 'dining table', '-', '-', 'toilet', '-', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '-', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush', '-'
        ]
    else:
        # VOC2012 lables
        table = [
            'aeroplane', 'bicycle', 'bird', 'boat' ,'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

    return table[label]


def draw_box(img, annot, is_coco=False, score_threshold=0.9, fontsize=0.7, split_acc=False):
    # Handle list inputs
    if isinstance(img, list) and isinstance(annot, list):
        return [draw_box(im, an, is_coco, score_threshold, fontsize, split_acc) 
                for im, an in zip(img, annot)]
    
    # Single image processing
    annot = annot.copy()
    img_tmp = np.array(img.permute(1,2,0).detach().cpu()*255).copy()

    if 'scores' in annot:
        mask = (annot['scores'] > score_threshold)
        annot['boxes'] = annot['boxes'] * mask.unsqueeze(1)
        annot['labels'] = annot['labels'] * mask

    def _draw_text_with_background(
        image,
        text,
        org,
        font_scale,
        text_color=(255, 255, 0),
        bg_color=(255, 0, 0),
        thickness=2,
        bg_alpha=0.5,
        y_offset=0
    ):
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x, y = org
        pad = 2

        x1 = max(0, x - pad - 2)
        y1 = max(0, y - text_h - baseline - pad + 4 + y_offset)
        x2 = min(image.shape[1] - 1, x + text_w + pad)
        y2 = min(image.shape[0] - 1, y + baseline + pad)

        if x2 >= x1 and y2 >= y1:
            roi = image[y1:y2 + 1, x1:x2 + 1]
            if roi.size > 0:
                overlay = np.full_like(roi, bg_color)
                image[y1:y2 + 1, x1:x2 + 1] = cv2.addWeighted(overlay, bg_alpha, roi, 1.0 - bg_alpha, 0)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, 2)

    if len(annot['boxes']) > 0:
        for idx in range(len(annot['boxes'])):
            x1, y1, x2, y2 = annot['boxes'][idx]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)                
            img_label = int(annot['labels'][idx])
            obj_label = convert2label(img_label-1, is_coco=is_coco)

            if obj_label == "-":
                continue  # Classes not used in detection

            if 'scores' in annot:
                obj_label_with_acc = '{}: {:.2f}'.format(obj_label, annot['scores'][idx])
            else:
                obj_label_with_acc = '{}'.format(obj_label)

            if (x1 < 0 or x2 > img.size(2) or y1 < 0 or y2 > img.size(1)) or (x1 < 10 and y1 < 10 and obj_label == 'tvmonitor'):
                # Bounding box exceeds image resolution — removing it improves visualization
                # The tvmonitor at the upper-left corner often leads to unstable detection results
                continue
            # if (y1 > 340) or (y1 < 23) or (obj_label == 'vase'):
            #     # RealPhoto60_36
            #     continue
            # if (x1 > 460):
            #     # GOPRO0884_11_00
            #     continue
            # if (x1 < 150):
            #     # EDTR-DEMO
            #     continue
            # if (x1 > 128):
            #     # GOPRO0884_11_00
            #     continue
            # if (obj_label == 'dining table'):
            #     # My-document
            #     continue
            
            img_tmp = cv2.rectangle(img_tmp, (x1, y1), (x2, y2), color=(255,0,0), thickness=2)
            if split_acc:
                # Split object class and accuracy terms in visualization
                _draw_text_with_background(img_tmp, obj_label, (x1+5, y1+20), fontsize)
                _draw_text_with_background(img_tmp, ':{:.2f}'.format(annot['scores'][idx]), (x1+5, y1+40), fontsize-0.2, y_offset=4)
            else:
                _draw_text_with_background(img_tmp, obj_label_with_acc, (x1+5, y1+20), fontsize)

    img_tmp = torch.Tensor(img_tmp).permute(2,0,1)/255.0

    return img_tmp


def _repeat_to_at_least(iterable, n):
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Args:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """

    def __init__(self, sampler, group_ids, batch_size):
        if not isinstance(sampler, Sampler):
            raise ValueError(f"sampler should be an instance of torch.utils.data.Sampler, but got sampler={sampler}")
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            # for the remaining batches, take first the buffers with the largest number
            # of elements
            for group_id, _ in sorted(buffer_per_group.items(), key=lambda x: len(x[1]), reverse=True):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        return len(self.sampler) // self.batch_size


def _compute_aspect_ratios_slow(dataset, indices=None):
    print(
        "Your dataset doesn't support the fast path for "
        "computing the aspect ratios, so will iterate over "
        "the full dataset and load every image instead. "
        "This might take some time..."
    )
    if indices is None:
        indices = range(len(dataset))

    class SubsetSampler(Sampler):
        def __init__(self, indices):
            self.indices = indices

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    sampler = SubsetSampler(indices)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=14,  # you might want to increase it for faster processing
        collate_fn=lambda x: x[0],
    )
    aspect_ratios = []
    with tqdm(total=len(dataset)) as pbar:
        for _i, (img, _) in enumerate(data_loader):
            pbar.update(1)
            height, width = img.shape[-2:]
            aspect_ratio = float(width) / float(height)
            aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_custom_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        height, width = dataset.get_height_and_width(i)
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_coco_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        img_info = dataset.coco.imgs[dataset.ids[i]]
        aspect_ratio = float(img_info["width"]) / float(img_info["height"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_voc_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        # this doesn't load the data into memory, because PIL loads it lazily
        width, height = Image.open(dataset.images[i]).size
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_subset_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))

    ds_indices = [dataset.indices[i] for i in indices]
    return compute_aspect_ratios(dataset.dataset, ds_indices)


def compute_aspect_ratios(dataset, indices=None):
    if hasattr(dataset, "get_height_and_width"):
        return _compute_aspect_ratios_custom_dataset(dataset, indices)

    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return _compute_aspect_ratios_coco_dataset(dataset, indices)

    if isinstance(dataset, torchvision.datasets.VOCDetection):
        return _compute_aspect_ratios_voc_dataset(dataset, indices)

    if isinstance(dataset, torch.utils.data.Subset):
        return _compute_aspect_ratios_subset_dataset(dataset, indices)

    # slow path
    return _compute_aspect_ratios_slow(dataset, indices)


def _quantize(x, bins):
    bins = copy.deepcopy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized

def create_aspect_ratio_groups(dataset, k=0, logger=None):
    aspect_ratios = compute_aspect_ratios(dataset)
    bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]
    groups = _quantize(aspect_ratios, bins)
    # count number of elements per group
    counts = np.unique(groups, return_counts=True)[1]
    fbins = [0] + bins + [np.inf]
    if logger is not None:
        logger(f"Aspect ratio quantization bins: {fbins}", print=False)
        logger(f"Count of instances per bin: {counts}", print=False)
    return groups


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # variable-sized images and annotations -> keep as lists
    ks = ["hq", "lq", "annot"]
    for k in ks:
        if k in batch[0]:
            out[k] = [b[k] for b in batch]

    # fixed-shape tensors -> stack
    ks = ["kernel1", "kernel2", "sinc_kernel"]
    for k in ks:
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch], dim=0)

    # strings -> list
    ks = ["txt", "filename"]
    for k in ks:
        if k in batch[0]:
            out[k] = [b[k] for b in batch]

    return out


def get_coco_api_from_dataset(dataset):
    # FIXME: This is... awful?
    # for _ in range(10):
    #     if isinstance(dataset, torchvision.datasets.CocoDetection):
    #         break
    #     if isinstance(dataset, torch.utils.data.Subset):
    #         dataset = dataset.dataset
    # if isinstance(dataset, torchvision.datasets.CocoDetection):
    #     return dataset.coco
    return convert_to_coco_api(dataset)


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img = ds[img_idx]["hq"]
        targets = ds[img_idx]["annot"]
        image_id = targets["image_id"]
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        if not isinstance(iou_types, (list, tuple)):
            raise TypeError(f"This constructor expects iou_types of type list or tuple, instead  got {type(iou_types)}")
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self, logger=None):
        results = {}
        for iou_type, coco_eval in self.coco_eval.items():
            if logger is not None:
                logger(f"IoU metric: {iou_type}", print=False)
                f = io.StringIO()
                with redirect_stdout(f):
                    coco_eval.summarize()
                for string in f.getvalue().split('\n'):
                    if not string == "":
                        logger(string, print=False)
                        if "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]" in string:
                            results["mAP@[0.5:0.95]"] = float(string.split("=")[-1]) * 100
                        elif "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]" in string:
                            results["mAP@0.5"] = float(string.split("=")[-1]) * 100
            else:
                print(f"IoU metric: {iou_type}")
                coco_eval.summarize()
        return results
            
    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        if iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(imgs):
    with redirect_stdout(io.StringIO()):
        imgs.evaluate()
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def sliding_windows(W, H, tile, stride):
    """Yield (x0, y0, x1, y1) windows that cover the WxH image with overlap."""
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    ys = list(range(0, max(H - tile, 0) + 1, stride))

    # ensure last tiles touch the right/bottom edge
    if xs and xs[-1] + tile < W:
        xs.append(W - tile)
    if ys and ys[-1] + tile < H:
        ys.append(H - tile)

    for y in ys:
        for x in xs:
            yield (x, y, x + tile, y + tile)


def move_boxes(boxes, dx, dy):
    """Shift boxes by (dx, dy). boxes: Tensor [N,4] in xyxy."""
    out = boxes.clone()
    out[:, [0, 2]] += dx
    out[:, [1, 3]] += dy
    return out
