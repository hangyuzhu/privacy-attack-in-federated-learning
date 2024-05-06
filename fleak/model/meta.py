import collections
from collections import OrderedDict
from itertools import repeat

import torch
import torch.nn.functional as F


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_pair = _ntuple(2, "_pair")


class MetaModel(torch.nn.Module):
    """ Meta Model for models built by torch.nn.Module

    Caution: 1) modules of model should be built in order
             2) modules not containing parameters are required to be constructed by nn.Module
    """

    def __init__(self, model):
        super(MetaModel, self).__init__()
        self.model = model
        self.parameters = OrderedDict(model.named_parameters())

    def forward(self, x, parameters=None):
        if parameters is None:
            return self.model(x)

        # construct an iterator
        param_gen = iter(parameters.values())
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                ext_weight = next(param_gen)
                if module.bias is not None:
                    ext_bias = next(param_gen)
                else:
                    ext_bias = None
                if module.padding_mode != "zeros":
                    x = F.conv2d(F.pad(x, module._reversed_padding_repeated_twice, mode=module.padding_mode),
                                 ext_weight, ext_bias, module.stride,
                                 _pair(0), module.dilation, module.groups)
                x = F.conv2d(x, ext_weight, ext_bias, module.stride,
                             module.padding, module.dilation, module.groups)
            elif isinstance(module, torch.nn.BatchNorm2d):
                if module.momentum is None:
                    exponential_average_factor = 0.0
                else:
                    exponential_average_factor = module.momentum

                if module.training and module.track_running_stats:
                    if module.num_batches_tracked is not None:
                        module.num_batches_tracked += 1
                        if module.momentum is None:  # use cumulative moving average
                            exponential_average_factor = 1.0 / float(module.num_batches_tracked)
                        else:  # use exponential moving average
                            exponential_average_factor = module.momentum

                ext_weight = next(param_gen)
                ext_bias = next(param_gen)
                x = F.batch_norm(
                    x,
                    running_mean=module.running_mean,
                    running_var=module.running_var,
                    weight=ext_weight,
                    bias=ext_bias,
                    training=module.training or not module.track_running_stats,
                    momentum=exponential_average_factor,
                    eps=module.eps
                )
            elif isinstance(module, torch.nn.Linear):
                lin_weights = next(param_gen)
                lin_bias = next(param_gen)
                x = F.linear(x, lin_weights, lin_bias)

            # for next(module.parameters(), None) is None
            elif isinstance(module, torch.nn.ReLU):
                x = F.relu(x, inplace=module.inplace)
            elif isinstance(module, torch.nn.MaxPool2d):
                x = F.max_pool2d(x, module.kernel_size, module.stride,
                                 module.padding, module.dilation, ceil_mode=module.ceil_mode,
                                 return_indices=module.return_indices)
            elif isinstance(module, torch.nn.Flatten):
                x = x.flatten(module.start_dim, module.end_dim)
            elif isinstance(module, torch.nn.Dropout):
                x = F.dropout(x, module.p, module.training, module.inplace)
            elif isinstance(module, torch.nn.Sequential):
                # Pass containers
                pass
            else:
                # Warn for other containers
                TypeError("Unexpected {}".format(type(module)))
        return x
