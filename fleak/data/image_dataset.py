import os
from typing import Any, Tuple

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets.folder import default_loader
from torchvision import datasets, transforms


class UnNormalize(torchvision.transforms.Normalize):
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


def load_mnist_dataset(data_dir):
    # ToTensor normalize images to 0 ~ 1
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        # transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    return train_dataset, test_dataset


def load_cifar10_dataset(data_dir, data_augment=False):
    if data_augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_eval)
    return train_dataset, test_dataset


def load_cifar100_dataset(data_dir, data_augment=False):
    if data_augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform_eval)
    return train_dataset, test_dataset


def load_tiny_imagenet_dataset(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    return train_dataset, test_dataset
