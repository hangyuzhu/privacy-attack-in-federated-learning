from ..model import MnistMLP
from ..model import CifarMLP
from ..model import MnistConvNet
from ..model import CifarConvNet
from ..model import ResNet18
from ..model import ResNet34
from ..model import FC2
from ..model import TinyImageNetVGG
from ..model import CifarVGG

from ..data.image_dataset import load_mnist_dataset
from ..data.image_dataset import load_cifar10_dataset
from ..data.image_dataset import load_cifar100_dataset
from ..data.image_dataset import load_tiny_imagenet_dataset


def get_model_options(dataset):
    model = {
        "mlp": MnistMLP if dataset == 'mnist' else CifarMLP,
        "cnn": MnistConvNet if dataset == 'mnist' else CifarConvNet,
        "resnet18": ResNet18,
        "resnet34": ResNet34,
        "fc2": FC2,
        "vgg16": TinyImageNetVGG if dataset == 'tiny_imagenet' else CifarVGG
    }
    return model


def get_dataset_options(dataset):
    if dataset == 'mnist':
        return load_mnist_dataset
    elif dataset == 'cifar10':
        return load_cifar10_dataset
    elif dataset == 'cifar100':
        return load_cifar100_dataset
    elif dataset == 'tiny_imagenet':
        return load_tiny_imagenet_dataset
    else:
        raise TypeError(f'{dataset} is not an expected dataset !')