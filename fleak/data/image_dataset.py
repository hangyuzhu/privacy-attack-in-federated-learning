import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader
from torchvision import datasets, transforms
from typing import Any, Tuple
import os


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class CombineDataset(Dataset):

    def __init__(self, trds, teds):
        if isinstance(trds.data, np.ndarray):
            assert isinstance(teds.data, np.ndarray)
            self.data = np.concatenate((trds.data, teds.data), axis=0)
        elif isinstance(trds.data, torch.Tensor):
            assert isinstance(teds.data, torch.Tensor)
            self.data = torch.cat((trds.data, teds.data), dim=0).numpy()
        else:
            raise TypeError('unexpected type {}'.format(type(trds.data)))
        if isinstance(trds.targets, np.ndarray):
            assert isinstance(teds.targets, np.ndarray)
            self.targets = np.concatenate((trds.targets, teds.targets), axis=0)
        elif isinstance(trds.targets, torch.Tensor):
            assert isinstance(teds.targets, torch.Tensor)
            self.targets = torch.cat((trds.targets, teds.targets), dim=0)
        elif isinstance(trds.targets, list):
            assert isinstance(teds.targets, list)
            self.targets = torch.tensor(trds.targets + teds.targets)
        else:
            raise TypeError('unexpected type {}'.format(type(trds.targets)))
        assert len(self.data) == len(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        return img, target


class ImageFolderCombineDataset(Dataset):

    def __init__(self, trds, teds):
        self.samples = trds.samples + teds.samples
        self.targets = trds.targets + teds.targets
        assert len(self.samples) == len(self.targets)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        return path, target


class CustomImageDataset(Dataset):

    def __init__(self, data, targets, transform=None, target_transform=None):
        self.data = data
        self.targets = targets
        assert len(self.data) == len(self.targets)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


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
        transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = datasets.MNIST(data_dir, train=False)
    return train_dataset, test_dataset, transform, transform


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
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True)
    return train_dataset, test_dataset, transform_train, transform_eval


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
    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True)
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True)
    return train_dataset, test_dataset, transform_train, transform_eval


def load_tiny_imagenet_dataset(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'))
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'))
    return train_dataset, test_dataset, transform, transform
