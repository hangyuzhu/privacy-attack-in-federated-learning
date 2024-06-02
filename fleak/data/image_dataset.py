import os
from typing import Any, Tuple

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader


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

"""

It should be noticed that, mean & std are probably sensitive to both training and
reconstruction outcomes !

IMAGE_MEAN & IMAGE_STD are used in the official implementation (pytorch) of image classification tasks
IMAGE_MEAN_GAN & IMAGE_STD_GAN are used in GAN for generating fake images with higher quality

"""
# mean & std applied in official pytorch training
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

# mean & std applied in GAN training
# Caution: using pretrained model for imagenet dataset
# apply the same mean & std in official pytorch training
# mean
IMAGE_MEAN_GAN = {
    "mnist": [0.5, ],
    "cifar10": [0.5, 0.5, 0.5],
    "cifar100": [0.5, 0.5, 0.5],
    "tiny_imagenet": [0.5, 0.5, 0.5],
    "imagenet": [0.485, 0.456, 0.406],

}
# std
IMAGE_STD_GAN = {
    "mnist": [0.5, ],
    "cifar10": [0.5, 0.5, 0.5],
    "cifar100": [0.5, 0.5, 0.5],
    "tiny_imagenet": [0.5, 0.5, 0.5],
    "imagenet": [0.229, 0.224, 0.225],
}


class UnNormalize(torchvision.transforms.Normalize):
    """ Inverse normalize operation """

    def __init__(self, mean, std, *args, **kwargs):
        new_mean = [-m/s for m, s in zip(mean, std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


class TinyImageNet(ImageFolder):

    def __init__(
        self,
        root: str,
        train=True,
        transform=None,
        target_transform=None
    ) -> None:
        subfolder = "train" if train else "val"
        root_sub = os.path.join(root, subfolder)

        if not os.path.exists(root):
            raise ValueError(
                "Dataset not found at {}. Please download it from {}.".format(
                    root, "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
                )
            )

        super().__init__(
            root=root_sub,
            transform=transform,
            target_transform=target_transform
        )


class ImageNet(ImageFolder):

    def __init__(
        self,
        root: str,
        train=True,
        transform=None,
        target_transform=None,
    ):
        subfolder = "train" if train else "val"
        root_sub = os.path.join(root, subfolder)

        if not os.path.exists(root):
            raise ValueError(f"Dataset not found at {root}.")

        super().__init__(
            root=root_sub,
            transform=transform,
            target_transform=target_transform,
        )


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


"""

Caution: IMAGE_MEAN_GAN & IMAGE_STD_GAN are selected here for the purpose of training classifiers
         with probably more stable performance and generating high-quality fake images

"""


def load_mnist_dataset(data_dir, normalize=True, data_augment=False):
    transform_train_list, transform_eval_list = [transforms.ToTensor()], [transforms.ToTensor()]
    if data_augment:
        transform_train_list += [transforms.RandomCrop(28, padding=4),
                                 transforms.RandomHorizontalFlip()]
    if normalize:
        dm, ds = IMAGE_MEAN_GAN["mnist"], IMAGE_STD_GAN["mnist"]
        transform_train_list += [transforms.Normalize(dm, ds)]
        transform_eval_list += [transforms.Normalize(dm, ds)]
    transform_train = transforms.Compose(transform_train_list)
    transform_eval = transforms.Compose(transform_eval_list)

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform_eval)
    return train_dataset, test_dataset


def load_cifar10_dataset(data_dir, normalize=True, data_augment=False):
    transform_train_list, transform_eval_list = [transforms.ToTensor()], [transforms.ToTensor()]
    if data_augment:
        transform_train_list += [transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip()]
    if normalize:
        dm, ds = IMAGE_MEAN_GAN["cifar10"], IMAGE_STD_GAN["cifar10"]
        transform_train_list += [transforms.Normalize(dm, ds)]
        transform_eval_list += [transforms.Normalize(dm, ds)]
    transform_train = transforms.Compose(transform_train_list)
    transform_eval = transforms.Compose(transform_eval_list)

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_eval)
    return train_dataset, test_dataset


def load_cifar100_dataset(data_dir, normalize=True, data_augment=False):
    transform_train_list, transform_eval_list = [transforms.ToTensor()], [transforms.ToTensor()]
    if data_augment:
        transform_train_list += [transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(15)]
    if normalize:
        dm, ds = IMAGE_MEAN_GAN["cifar100"], IMAGE_STD_GAN["cifar100"]
        transform_train_list += [transforms.Normalize(dm, ds)]
        transform_eval_list += [transforms.Normalize(dm, ds)]
    transform_train = transforms.Compose(transform_train_list)
    transform_eval = transforms.Compose(transform_eval_list)

    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform_eval)
    return train_dataset, test_dataset


def load_tiny_imagenet_dataset(data_dir, normalize=True, data_augment=False):
    transform_train_list, transform_eval_list = [transforms.ToTensor()], [transforms.ToTensor()]
    if data_augment:
        transform_train_list += [transforms.RandomCrop(64),
                                 transforms.RandomHorizontalFlip()]
    if normalize:
        dm, ds = IMAGE_MEAN_GAN["tiny_imagenet"], IMAGE_STD_GAN["tiny_imagenet"]
        transform_train_list += [transforms.Normalize(dm, ds)]
        transform_eval_list += [transforms.Normalize(dm, ds)]
    transform_train = transforms.Compose(transform_train_list)
    transform_eval = transforms.Compose(transform_eval_list)

    train_dataset = TinyImageNet(data_dir, train=True, transform=transform_train)
    test_dataset = TinyImageNet(data_dir, train=False, transform=transform_eval)
    return train_dataset, test_dataset


def load_imagenet_dataset(data_dir, normalize=True, data_augment=False):
    transform_train_list, transform_eval_list = [transforms.ToTensor()], [transforms.ToTensor()]
    if data_augment:
        transform_train_list += [transforms.RandomCrop(224),
                                 transforms.RandomHorizontalFlip()]
    else:
        transform_train_list += [transforms.Resize(size=(256, 256)),
                                 transforms.CenterCrop(size=(224, 224))]
    transform_eval_list += [transforms.Resize(size=(256, 256)),
                            transforms.CenterCrop(size=(224, 224))]
    if normalize:
        dm, ds = IMAGE_MEAN_GAN["imagenet"], IMAGE_STD_GAN["imagenet"]
        transform_train_list += [transforms.Normalize(dm, ds)]
        transform_eval_list += [transforms.Normalize(dm, ds)]
    transform_train = transforms.Compose(transform_train_list)
    transform_eval = transforms.Compose(transform_eval_list)

    train_dataset = ImageNet(data_dir, train=True, transform=transform_train)
    test_dataset = ImageNet(data_dir, train=False, transform=transform_eval)
    return train_dataset, test_dataset
