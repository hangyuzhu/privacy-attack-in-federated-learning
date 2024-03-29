import math
import torch
import torch.nn as nn
from scipy.stats import laplace
from statistics import NormalDist
from .neural_network import ResNet18, MnistConvNet, CifarConvNet


class ImprintBlock(torch.nn.Module):
    structure = "cumulative"

    def __init__(self, data_size, num_bins, connection="linear", gain=1e-3, linfunc="fourier", mode=0):
        """
        data_size is the length of the input data
        num_bins is how many "paths" to include in the model
        connection is how this block should coonect back to the input shape (optional)

        linfunc is the choice of linear query function ('avg', 'fourier', 'randn', 'rand').
        If linfunc is fourier, then the mode parameter determines the mode of the DCT-2 that is used as linear query.
        """
        super().__init__()
        self.data_size = data_size
        self.num_bins = num_bins
        self.linear0 = torch.nn.Linear(data_size, num_bins)

        self.bins = self._get_bins(linfunc)
        with torch.no_grad():
            self.linear0.weight.data = self._init_linear_function(linfunc, mode) * gain
            self.linear0.bias.data = self._make_biases() * gain

        self.connection = connection
        if connection == "linear":
            self.linear2 = torch.nn.Linear(num_bins, data_size)
            with torch.no_grad():
                self.linear2.weight.data = torch.ones_like(self.linear2.weight.data) / gain
                self.linear2.bias.data -= torch.as_tensor(self.bins).mean()

        self.relu = torch.nn.ReLU()

    @torch.no_grad()
    def _init_linear_function(self, linfunc="fourier", mode=0):
        K, N = self.num_bins, self.data_size
        if linfunc == "avg":
            weights = torch.ones_like(self.linear0.weight.data) / N
        elif linfunc == "fourier":
            weights = torch.cos(math.pi / N * (torch.arange(0, N) + 0.5) * mode).repeat(K, 1) / N * max(mode, 0.33) * 4
            # don't ask about the 4, this is WIP
            # nonstandard normalization
        elif linfunc == "randn":
            weights = torch.randn(N).repeat(K, 1)
            std, mu = torch.std_mean(weights[0])  # Enforce mean=0, std=1 with higher precision
            weights = (weights - mu) / std / math.sqrt(N)  # Move to std=1 in output dist
        elif linfunc == "rand":
            weights = torch.rand(N).repeat(K, 1)  # This might be a terrible idea haven't done the math
            std, mu = torch.std_mean(weights[0])  # Enforce mean=0, std=1
            weights = (weights - mu) / std / math.sqrt(N)  # Move to std=1 in output dist
        else:
            raise ValueError(f"Invalid linear function choice {linfunc}.")

        return weights

    def _get_bins(self, linfunc="avg"):
        bins = []
        mass_per_bin = 1 / (self.num_bins)
        bins.append(-10)  # -Inf is not great here, but NormalDist(mu=0, sigma=1).cdf(10) approx 1
        for i in range(1, self.num_bins):
            if "fourier" in linfunc:
                bins.append(laplace(loc=0.0, scale=1 / math.sqrt(2)).ppf(i * mass_per_bin))
            else:
                bins.append(NormalDist().inv_cdf(i * mass_per_bin))
        return bins

    def _make_biases(self):
        new_biases = torch.zeros_like(self.linear0.bias.data)
        for i in range(new_biases.shape[0]):
            new_biases[i] = -self.bins[i]
        return new_biases

    def forward(self, x):
        x_in = x
        x = self.linear0(x)
        x = self.relu(x)
        if self.connection == "linear":
            output = self.linear2(x)
        elif self.connection == "cat":
            output = torch.cat([x, x_in[:, self.num_bins :]], dim=1)
        elif self.connection == "softmax":
            s = torch.softmax(x, dim=1)[:, :, None]
            output = (x_in[:, None, :] * s).sum(dim=1)
        else:
            output = x_in + x.mean(dim=1, keepdim=True)
        return output


class RobinFed(nn.Module):

    def __init__(self, data_size, num_bins, num_class, shape_img, dataset='cifar10', model_type="resnet18"):
        super(RobinFed, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_block = ImprintBlock(data_size, num_bins)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=tuple(shape_img[1:]))
        self.base = self._get_base(model_type, dataset, num_class)

    def _get_base(self, model_type, dataset, num_class):
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
        x = self.linear_block(x)
        x = self.unflatten(x)
        x = self.base(x)
        return x
