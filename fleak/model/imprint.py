import math
import torch
import torch.nn as nn
from scipy.stats import laplace
from statistics import NormalDist


class ImprintModel(nn.Module):
    """Imprint model

    Linear imprint model + original model

    """

    def __init__(self, num_classes, base_module, input_shape, num_bins=100):
        super(ImprintModel, self).__init__()
        self.flatten = nn.Flatten()
        self.imprint_block = ImprintBlock(input_shape, num_bins)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=input_shape[1:])
        self.base = base_module(num_classes)

    def forward(self, x):
        # imprint parts
        x = self.flatten(x)
        x = self.imprint_block(x)
        x = self.unflatten(x)
        # normal forward pass
        x = self.base(x)
        return x


class ImprintBlock(nn.Module):

    def __init__(self, input_shape, num_bins=100, connection="linear", gain=1e-3, linfunc="fourier", mode=0):
        """Imprint block (works like an autoencoder)

        According to the original paper, replacing the normal distribution by a Laplacian distribution
        does oes improve the accuracy slight

        :param input_shape: the shape of input data containing batch size
        :param num_bins: number of bins
        :param gain: init gain
        :param linfunc: the choice of linear query function ('avg', 'fourier', 'randn', 'rand').
                        If linfunc is fourier, then the mode parameter
                        determines the mode of the DCT-2 that is used as linear query.
        :param mode: default to be 0
        """
        super().__init__()
        self.input_size = math.prod(input_shape[1:])
        self.num_bins = num_bins
        self.linear0 = torch.nn.Linear(self.input_size, num_bins)

        self.bins = self._get_bins(linfunc)
        with torch.no_grad():
            self.linear0.weight.data = self._init_linear_function(linfunc, mode) * gain
            self.linear0.bias.data = self._make_biases() * gain

        self.connection = connection
        if self.connection == "linear":
            self.linear1 = torch.nn.Linear(num_bins, self.input_size)
            with torch.no_grad():
                self.linear1.weight.data = torch.ones_like(self.linear1.weight.data) / gain
                self.linear1.bias.data -= torch.as_tensor(self.bins).mean()

        self.relu = torch.nn.ReLU()

    @torch.no_grad()
    def _init_linear_function(self, linfunc="fourier", mode=0):
        K, N = self.num_bins, self.input_size
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
            output = self.linear1(x)
        elif self.connection == "cat":
            output = torch.cat([x, x_in[:, self.num_bins:]], dim=1)
        elif self.connection == "softmax":
            s = torch.softmax(x, dim=1)[:, :, None]
            output = (x_in[:, None, :] * s).sum(dim=1)
        else:
            output = x_in + x.mean(dim=1, keepdim=True)
        return output


def _init_fourier_weight(tensor, mode=0):
    K, N = tensor.shape[0], tensor.shape[1]
    weights = torch.cos(math.pi / N * (torch.arange(0, N) + 0.5) * mode).repeat(K, 1) / N * max(mode, 0.33) * 4
    with torch.no_grad():
        tensor.copy_(weights)


def _init_laplacian_bias(tensor):
    num_bins = len(tensor)
    bins = []
    mass_per_bin = 1 / num_bins
    bins.append(-10)  # -Inf is not great here, but NormalDist(mu=0, sigma=1).cdf(10) approx 1
    for i in range(1, num_bins):
        bins.append(laplace(loc=0.0, scale=1 / math.sqrt(2)).ppf(i * mass_per_bin))

    new_biases = torch.zeros_like(tensor)
    for i in range(num_bins):
        new_biases[i] = -bins[i]
    with torch.no_grad():
        tensor.copy_(new_biases)
