import os
from typing import Any, Tuple

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets.folder import default_loader
from torchvision import datasets, transforms


N_CLASSES = {
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "tiny_imagenet": 200,
    "imagenet": 1000,
}

# channel first for pytorch
IMAGE_SHAPE = {
    "mnist": [1, 28, 28],
    "cifar10": [3, 32, 32],
    "cifar100": [3, 32, 32],
    "tiny_imagenet": [3, 64, 64],
    "imagenet": [3, 224, 224],
}

# mean
IMAGE_MEAN = {
    "mnist": [0.1307, ],
    "cifar10": [0.4914, 0.4822, 0.4465],
    "cifar100": [0.5071, 0.4867, 0.4408],
    "tiny_imagenet": [0.485, 0.456, 0.406],
    "imagenet": [0.485, 0.456, 0.406],
}

# std
IMAGE_STD = {
    "mnist": [0.3081, ],
    "cifar10": [0.2023, 0.1994, 0.2010],
    "cifar100": [0.2675, 0.2565, 0.2761],
    "tiny_imagenet": [0.229, 0.224, 0.225],
    "imagenet": [0.229, 0.224, 0.225],
}


class UnNormalize(torchvision.transforms.Normalize):
    """ Inverse normalize operation """

    def __init__(self, mean, std, *args, **kwargs):
        new_mean = [-m/s for m, s in zip(mean, std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class ImageFolderDataset(Dataset):

    def __init__(self, samples, loader=default_loader, transform=None, target_transform=None):
        self.samples = samples
        self.targets = torch.tensor([s[1] for s in samples])
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def load_mnist_dataset(data_dir, dm=None, ds=None):
    if dm is None:
        dm = IMAGE_MEAN["mnist"]
    if ds is None:
        ds = IMAGE_STD["mnist"]
    # ToTensor normalize images to 0 ~ 1
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dm, ds)
    ])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    return train_dataset, test_dataset


def load_cifar10_dataset(data_dir, dm=None, ds=None, data_augment=False):
    if dm is None:
        dm = IMAGE_MEAN["cifar10"]
    if ds is None:
        ds = IMAGE_STD["cifar10"]
    if data_augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dm, ds),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dm, ds),
        ])

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dm, ds),
    ])
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_eval)
    return train_dataset, test_dataset


def load_cifar100_dataset(data_dir, dm=None, ds=None, data_augment=False):
    if dm is None:
        dm = IMAGE_MEAN["cifar100"]
    if ds is None:
        ds = IMAGE_STD["cifar100"]
    if data_augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(dm, ds)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dm, ds)
        ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dm, ds)
    ])
    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform_eval)
    return train_dataset, test_dataset


def load_tiny_imagenet_dataset(data_dir, dm=None, ds=None):
    if dm is None:
        dm = IMAGE_MEAN["tiny_imagenet"]
    if ds is None:
        ds = IMAGE_STD["tiny_imagenet"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dm, ds),
    ])
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    return train_dataset, test_dataset
