# Generating degraded validation sets

This document describes how to generate the degraded validation sets yourself,
instead of downloading our pre-generated versions.

> [!NOTE]
> The degradation pipeline involves random sampling. While we fix the random seed,
> results may still differ slightly across different GPU or library environments.
> To exactly reproduce the numbers in our paper, we recommend using the
> pre-generated validation sets linked in the README.

## 1. Download the original datasets

| Task | Dataset | Download |
|:---:|:---:|:---:|
| Classification | CUB-200-2011 | [Kaggle](https://www.kaggle.com/datasets/wenewone/cub2002011?select=CUB_200_2011) |
| Segmentation / Detection | VOC2012 | [Kaggle](https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset) |
| Detection | COCO2017 | [Official](https://cocodataset.org/#download) |
| OCR | MJSynth | [Dropbox](https://www.dropbox.com/scl/fo/zf04eicju8vbo4s6wobpq/ALAXXq2iwR6wKJyaybRmHiI?rlkey=2rywtkyuz67b20hk58zkfhh2r&e=2&dl=0) |

**CUB-200-2011 / VOC2012**: On the Kaggle page, click "Download" and select "Download dataset as zip" (login required). Place the downloaded `archive.zip` in `datasets/source/` and run:

```shell
python preprocess/cub200.py --remove_archive  # CUB200
python preprocess/voc2012.py --remove_archive  # VOC2012
```

These scripts extract the zip and reorganize the dataset to match our code.

:warning: Both Kaggle downloads are named `archive.zip`, so process one dataset at a time.

**COCO2017**: Unzip and place under `datasets/source/`, e.g., `datasets/source/COCO/val2017`.

**MJSynth**: Download `data_lmdb_release.zip` and unzip to `datasets/source/data_lmdb_release`. The link is provided by [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here).

## 2. Generate the degraded validation sets

```shell
# CUB200
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/classification/cub200/data_gen/realesrgan.yaml

# VOC2012 (segmentation)
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/segmentation/voc2012/data_gen/realesrgan.yaml

# VOC2012 (detection)
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/detection/voc2012/data_gen/realesrgan.yaml

# COCO (detection; used in supplementary experiments)
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/detection/coco/data_gen/realesrgan.yaml

# MJSynth (OCR)
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/optical_character_recognition/mj/data_gen/realesrgan.yaml
```

Generated images are saved under the corresponding `experiments/` directory,
e.g., `experiments/classification/cub200/data/realesrgan/images` for classification.

## 3. Place the generated images

Move the generated images to `datasets/source/` so that the evaluation configs can find them.
For example, for classification:

```shell
mv experiments/classification/cub200/data/realesrgan/images datasets/source/CUB200/val-realesrgan
```
