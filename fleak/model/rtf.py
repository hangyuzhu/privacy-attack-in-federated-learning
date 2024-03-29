import math
import torch
import torch.nn as nn
from scipy.stats import laplace
from .neural_network import ResNet18,MnistConvNet,CifarConvNet



class RobinFed(nn.Module):
    def __init__(self, data_size, num_bins, num_class, shape_img, dataset='cifar10', model_type="resnet18", gain=1e-3, mode=0):
        super(RobinFed, self).__init__()
        self.num_bins = num_bins
        self.data_size = data_size
        self.flatten = nn.Flatten()
        self.linear0 = nn.Linear(data_size, num_bins)
        self.bins = self._get_bins()
        with torch.no_grad():
            self.linear0.weight.data = self._init_linear_function(mode) * gain
            self.linear0.bias.data = self._make_biases() * gain
        self.linear1 = nn.Linear(num_bins, data_size)
        with torch.no_grad():
            self.linear1.weight.data = torch.ones_like(self.linear1.weight.data) / gain
            self.linear1.bias.data -= torch.as_tensor(self.bins).mean()
        self.nonlin = torch.nn.ReLU()
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=tuple(shape_img[1:]))
        self.base = self._get_base(model_type, dataset, num_class)

    def _get_bins(self):
        bins = []
        mass_per_bin = 1 / (self.num_bins)
        bins.append(-10)
        for i in range(1, self.num_bins):
            bins.append(laplace(loc=0.0, scale=1 / math.sqrt(2)).ppf(i * mass_per_bin))
        return bins

    def _init_linear_function(self, mode=0):
        K, N = self.num_bins, self.data_size
        weights = torch.cos(math.pi / N * (torch.arange(0, N) + 0.5) * mode).repeat(K, 1) / N * max(mode, 0.33) * 4
        return weights

    def _make_biases(self):
        new_biases = torch.zeros_like(self.linear0.bias.data)
        for i in range(new_biases.shape[0]):
            new_biases[i] = -self.bins[i]
        return new_biases


    def _get_base(self,model_type, dataset, num_class):
        if model_type == "resnet18":
            return ResNet18(num_class)
        elif model_type == "cnn" and dataset == 'cifar10':
            return CifarConvNet(num_class)
        elif model_type == "cnn" and dataset == 'mnist':
            return MnistConvNet(num_class)
        else:
            print('The model type is error!')


    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(self.linear0(x))
        x = self.unflatten(x)
        x = self.base(x)
        return x

