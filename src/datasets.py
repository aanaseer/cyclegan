"""Implements the ImageDataset class to create a dataset for training/testing."""

from __future__ import annotations

import os

import glob

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision


class ImageDataset(Dataset):
    """Creates a data set to be used for either training or testing."""
    def __init__(self,
                 path: str,
                 kind: str,
                 transform: torchvision.transforms.transforms.Compose = None) -> None:
        """Initialises the data set.

        Args:
            path: Path to the data set.
            kind: Either "train" or "test" to decide which data files to create the data set with.
            transform: Composition of pytorch transforms to transform the data with.
        """
        self.transform = transform
        self.path = path
        self.kind = kind
        self.dataA = glob.glob(os.path.join(self.path, f"{self.kind}A", "*.*"))
        self.dataB = glob.glob(os.path.join(self.path, f"{self.kind}B", "*.*"))
        self.dataA_len = len(self.dataA)
        self.dataB_len = len(self.dataB)
        self.length = min(self.dataA_len, self.dataB_len)

    def __getitem__(self,
                    item: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets individual items from the data set.

        Args:
            item: Index to use to retrieve an item.

        Returns:
            A tuple with two torch Tensors of the images.
        """
        imageA = self.transform(Image.open(self.dataA[item % self.dataA_len]).convert('RGB'))
        imageB = self.transform(Image.open(self.dataB[item % self.dataB_len]).convert('RGB'))

        return imageA, imageB

    def __len__(self):
        return self.length
