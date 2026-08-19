#!/bin/bash
# =============================================================================
# NOLA-IR: Full list of inference and training commands
# =============================================================================
# This file is a command reference, not meant to be executed at once.
# Copy and run individual lines as needed.
#
# Options:
#   --save-img   Save result images during evaluation
#   --eval-only  Evaluate with a training config
#
# Config naming:
#   *_hq            Task model trained on HQ images (upper bound)
#   *_lq            Task model trained on LQ images (lower bound)
#   *_res-only      Restoration model pre-training
#   *_res-fix       Task model trained on frozen-restoration outputs
#   *_diffbir-only  DiffBIR restoration pre-training
#   *_diffbir-fix   Task model trained on frozen-DiffBIR outputs
#   *_sr4ir         SR4IR (comparison method)
#   *_edtr          EDTR (comparison method)
#   *_nola-ir       NOLA-IR (ours)
#   *_nola-ir-tpgan NOLA-IR + TPGAN (ours, final)
# =============================================================================


# =============================================================================
# Inference
# =============================================================================
## Classification (CUB200)
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/classification/cub200/eval/000_hq.yaml
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/classification/cub200/eval/001_lq.yaml
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/classification/cub200/eval/009_nola-ir-tpgan.yaml

## Segmentation (VOC2012)
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/segmentation/voc2012/eval/000_hq.yaml
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/segmentation/voc2012/eval/001_lq.yaml
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/segmentation/voc2012/eval/009_nola-ir-tpgan.yaml

## Detection (VOC2012)
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/detection/voc2012/eval/000_hq.yaml
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/detection/voc2012/eval/001_lq.yaml
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/detection/voc2012/eval/009_nola-ir-tpgan.yaml

## OCR (MJSynth)
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/optical_character_recognition/mj/eval/000_hq.yaml
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/optical_character_recognition/mj/eval/001_lq.yaml
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/optical_character_recognition/mj/eval/006_nola-ir-tpgan.yaml


# =============================================================================
# Training
# =============================================================================
## Classification (CUB200)
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 24177 run.py --config configs/classification/cub200/train/000_hq.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 24177 run.py --config configs/classification/cub200/train/001_lq.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/classification/cub200/train/002_res-only.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 24177 run.py --config configs/classification/cub200/train/003_res-fix.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/classification/cub200/train/004_diffbir-only.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/classification/cub200/train/005_diffbir-fix.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/classification/cub200/train/006_sr4ir.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/classification/cub200/train/007_edtr.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/classification/cub200/train/008_nola-ir.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/classification/cub200/train/009_nola-ir-tpgan.yaml

## Segmentation (VOC2012)
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 24177 run.py --config configs/segmentation/voc2012/train/000_hq.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 24177 run.py --config configs/segmentation/voc2012/train/001_lq.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/segmentation/voc2012/train/002_res-only.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 24177 run.py --config configs/segmentation/voc2012/train/003_res-fix.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/segmentation/voc2012/train/004_diffbir-only.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/segmentation/voc2012/train/005_diffbir-fix.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/segmentation/voc2012/train/006_sr4ir.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/segmentation/voc2012/train/007_edtr.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/segmentation/voc2012/train/008_nola-ir.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/segmentation/voc2012/train/009_nola-ir-tpgan.yaml

## Detection (VOC2012)
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 24177 run.py --config configs/detection/voc2012/train/000_hq.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 24177 run.py --config configs/detection/voc2012/train/001_lq.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/detection/voc2012/train/002_res-only.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 24177 run.py --config configs/detection/voc2012/train/003_res-fix.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/detection/voc2012/train/004_diffbir-only.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/detection/voc2012/train/005_diffbir-fix.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/detection/voc2012/train/006_sr4ir.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/detection/voc2012/train/007_edtr.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/detection/voc2012/train/008_nola-ir.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/detection/voc2012/train/009_nola-ir-tpgan.yaml

## Detection (COCO)
# NOTE: Requires the COCO-pretrained FasterRCNN (ResNet50-FPN) weights.
# Reference: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html
# Prepare them with torchvision:
#   python -c "
#   from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
#   import torch
#   m = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
#   torch.save(m.state_dict(), 'weights/FasterRCNN_ResNet50_FPN_V2_Weights_COCO_V1.pt')"
#
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/detection/coco/train/000_res-only.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/detection/coco/train/001_nola-ir.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/detection/coco/train/002_nola-ir-tpgan.yaml

## OCR (MJSynth)
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 24177 run.py --config configs/optical_character_recognition/mj/train/000_hq.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 24177 run.py --config configs/optical_character_recognition/mj/train/001_lq.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/optical_character_recognition/mj/train/002_res-only.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 24177 run.py --config configs/optical_character_recognition/mj/train/003_res-fix.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/optical_character_recognition/mj/train/004_sr4ir.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/optical_character_recognition/mj/train/005_nola-ir.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 24177 run.py --config configs/optical_character_recognition/mj/train/006_nola-ir-tpgan.yaml
