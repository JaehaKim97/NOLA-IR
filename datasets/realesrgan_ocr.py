import os
import re
import random
from typing import Dict, Optional, Sequence

import lmdb
import numpy as np
import six
import torch
from glob import glob
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from datasets.realesrgan import RealESRGANDegradationMixin


def RealesrganOCRDataset(**kwargs):
    dataset_type = kwargs.pop("format", "lmdb").lower()
    if dataset_type == "lmdb":
        data_paths = kwargs.pop("data_paths")
        dataset_list = []
        for data_path in data_paths:
            dataset = RealesrganLMDBDataset(data_path, **kwargs)
            dataset_list.append(dataset)
        concat_dataset = ConcatDataset(dataset_list)
        concat_dataset.root = [d.root for d in dataset_list]
        return concat_dataset
    if dataset_type == "img":        
        return RealesrganIMGDataset(**kwargs)
    raise ValueError(f"Unknown dataset_type: {dataset_type} (expected 'lmdb' or 'img')")


class RealesrganLMDBDataset(RealESRGANDegradationMixin, Dataset):
    def __init__(
        self,
        root: str,
        character: str,
        case_sensitive: bool,
        gt_size: int,
        resize_range: Sequence[float],
        out_size: int,
        batch_max_length: int,
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
        data_filtering_off: bool = False,
        data_length: int = -1,
        return_file_name: bool = False,
        random_select_prob: float = 0.0,
        mode: str = None,
    ) -> "RealesrganLMDBDataset":
        self.root = root
        self.character = str(character)
        self.case_sensitive = bool(case_sensitive)
        self.gt_size = gt_size
        self.resize_range = resize_range
        self.out_size = out_size
        self.data_length = int(data_length)
        self.return_file_name = bool(return_file_name)

        # real-esrgan degradation
        self.init_degradation(
            blur_kernel_size, kernel_list, kernel_prob, blur_sigma, betag_range, betap_range, sinc_prob,
            blur_kernel_size2, kernel_list2, kernel_prob2, blur_sigma2, betag_range2, betap_range2, sinc_prob2,
            final_sinc_prob,
        )

        # prepare lmdb for loading
        self.env = None  # lazy loading for multiprocessing
        env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not env:
            print(f'Cannot create lmdb from {root}')
            raise NotImplementedError
        with env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            if data_filtering_off:
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')
                    if len(label) > batch_max_length:
                        continue
                    out_of_char = f'[^{character}]'
                    if re.search(out_of_char, label.lower()):
                        continue
                    self.filtered_index_list.append(index)
                
                if data_length > 0:
                    self.filtered_index_list = self.filtered_index_list[:data_length]
                if random_select_prob > 0.0:
                    self.filtered_index_list = [
                        index for index in self.filtered_index_list if random.random() < random_select_prob
                    ]
                if (mode is not None):
                    if mode == "train":
                        self.filtered_index_list = [
                            idx for i, idx in enumerate(self.filtered_index_list) if i % 100 != 1
                        ]
                    elif mode == "valid":
                        self.filtered_index_list = [
                            idx for i, idx in enumerate(self.filtered_index_list) if i % 100 == 1
                        ]
                self.nSamples = len(self.filtered_index_list)

    def __getstate__(self):
        d = dict(self.__dict__)
        d["env"] = None
        return d

    def _get_env(self):
        if self.env is None:
            self.env = lmdb.open(self.root, max_readers=512, readonly=True, lock=False, readahead=False, meminit=False)
        return self.env

    def __len__(self) -> int:
        return self.data_length if self.data_length > 0 else self.nSamples

    def load_items(self, txn, img_key: str, label_key: str, max_retry: int=5) -> Optional[np.ndarray]:
        img_buf = txn.get(img_key.encode())
        buf = six.BytesIO()
        buf.write(img_buf)
        buf.seek(0)
        try:
            image = Image.open(buf).convert('RGB')
        except IOError:
            print(f'Corrupted image for {img_key}')
            raise NotImplementedError("Error occurs in data loading process")

        image = image.resize(self.gt_size, Image.BICUBIC)

        # augmentation
        if self.resize_range is not None:
            r = random.uniform(*self.resize_range)
            new_size = tuple(round(dim * r) for dim in image.size)
            image = image.resize(new_size, Image.BICUBIC)
            if r > 1.0:
                ch = random.randint(0, image.height - self.out_size[1])
                cw = random.randint(0, image.width - self.out_size[0])
                image = np.array(image)[ch:ch+self.out_size[1], cw:cw+self.out_size[0]]
            elif r < 1.0:
                ph = random.randint(0, self.out_size[1] - image.height)
                pw = random.randint(0, self.out_size[0] - image.width)
                padding = ((ph, self.out_size[1] - image.height - ph), (pw, self.out_size[0] - image.width - pw), (0, 0))
                image = np.pad(image, padding, mode='edge')
        else:
            image = np.array(image)

        label = txn.get(label_key.encode()).decode('utf-8')
        if not self.case_sensitive:
            label = label.lower()
        out_of_char = f'[^{self.character}]'
        label = re.sub(out_of_char, '', label)

        # hwc, rgb, 0,255, uint8
        return image, label

    def __getitem__(self, index: int, max_retry: int=5) -> Dict[str, torch.Tensor]:
        index = index % len(self.filtered_index_list)
        index = self.filtered_index_list[index]
        
        env = self._get_env()  # lazty loading for multiprocessing
        with env.begin(write=False) as txn:
            img_key = f'image-{index:09d}'
            label_key = f'label-{index:09d}'
            img_gt, label = self.load_items(txn, img_key, label_key)
            prompt = ""

        # shape: (c, h, w); channel order: RGB; image range: [0, 1], float32.
        img_hq = torch.from_numpy((img_gt / 255.0).astype(np.float32).transpose(2, 0, 1).copy()).float()
        kernel, kernel2, sinc_kernel = self.sample_degradation_kernels()

        data = {
            "hq": img_hq,
            "kernel1": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
            "txt": prompt,
            "label": label,
        }
        if self.return_file_name:
            data["filename"] = os.path.split(self.root)[1] + "-" + img_key

        return data


class RealesrganIMGDataset(RealESRGANDegradationMixin, Dataset):
    def __init__(
        self,
        root: str,
        character: str,
        case_sensitive: bool,
        gt_size: int,
        resize_range: Sequence[float],
        out_size: int,
        batch_max_length: int,
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
        random_select_prob: float = 0.0,
    ) -> "RealesrganIMGDataset":
        self.root = root
        self.character = str(character)
        self.case_sensitive = bool(case_sensitive)
        self.gt_size = gt_size
        self.resize_range = resize_range
        self.out_size = out_size
        self.data_length = int(data_length)
        self.return_file_name = bool(return_file_name)

        # real-esrgan degradation
        self.init_degradation(
            blur_kernel_size, kernel_list, kernel_prob, blur_sigma, betag_range, betap_range, sinc_prob,
            blur_kernel_size2, kernel_list2, kernel_prob2, blur_sigma2, betag_range2, betap_range2, sinc_prob2,
            final_sinc_prob,
        )

        self.images = glob(os.path.join(root, '*.png'))

    def __len__(self) -> int:
        return self.data_length if self.data_length > 0 else len(self.images)

    def load_items(self, image_path: str, max_retry: int=5) -> Optional[np.ndarray]:
        try:
            image = Image.open(image_path).convert('RGB')  # for color image
        except IOError:
            print(f'Corrupted image for {image_path}')
            raise NotImplementedError("Error occurs in data loading process")

        image = image.resize(self.gt_size, Image.BICUBIC)

        # augmentation
        if self.resize_range is not None:
            r = random.uniform(*self.resize_range)
            new_size = tuple(round(dim * r) for dim in image.size)
            image = image.resize(new_size, Image.BICUBIC)
            if r > 1.0:
                ch = random.randint(0, image.height - self.out_size[1])
                cw = random.randint(0, image.width - self.out_size[0])
                image = np.array(image)[ch:ch+self.out_size[1], cw:cw+self.out_size[0]]
            elif r < 1.0:
                ph = random.randint(0, self.out_size[1] - image.height)
                pw = random.randint(0, self.out_size[0] - image.width)
                padding = ((ph, self.out_size[1] - image.height - ph), (pw, self.out_size[0] - image.width - pw), (0, 0))
                image = np.pad(image, padding, mode='edge')
        else:
            image = np.array(image)

        label = os.path.splitext(image_path)[0].split("/")[-1].split("-")[0]
        if not self.case_sensitive:
            label = label.lower()
        out_of_char = f'[^{self.character}]'
        label = re.sub(out_of_char, '', label)

        # hwc, rgb, 0,255, uint8
        return image, label

    def __getitem__(self, index: int, max_retry: int=5) -> Dict[str, torch.Tensor]:
        assert index <= len(self), 'index range error'
        image_path = self.images[index]
        
        try:
            img_gt, label = self.load_items(image_path)
            prompt = ""
        except IOError:
            print(f'Corrupted image for {index}')
            raise NotImplementedError("Error occurs in data loading process")

        # shape: (c, h, w); channel order: RGB; image range: [0, 1], float32.
        img_hq = torch.from_numpy((img_gt / 255.0).astype(np.float32).transpose(2, 0, 1).copy()).float()
        kernel, kernel2, sinc_kernel = self.sample_degradation_kernels()

        data = {
            "hq": img_hq,
            "kernel1": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
            "txt": prompt,
            "label": label,
        }
        if self.return_file_name:
            data["filename"] = image_path

        return data
