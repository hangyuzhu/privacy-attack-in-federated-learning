import collections
from collections import OrderedDict
from itertools import repeat

import torch
import torch.nn.functional as F

from .neural_network import BasicBlock


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

    Only support parts of pytorch models
    You can insert more nn.Module for further extension

    Caution: 1) modules of model should be built in order
             2) modules not containing parameters are required to be constructed by nn.Module
    """

    def __init__(self, model):
        super(MetaModel, self).__init__()
        self.model = model
        self.parameters = OrderedDict(model.named_parameters())

    def forward(self, x, parameters=None):
        """

        Caution: shortcut connections are only valid when
        the corresponding nn.Module attributes of ops are named as "shortcut"

        """
        if parameters is None:
            return self.model(x)

        # construct an iterator
        param_gen = iter(parameters.values())
        # buffer stored for shortcut connection
        x_in = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                ext_weight = next(param_gen)
                if module.bias is not None:
                    ext_bias = next(param_gen)
                else:
                    ext_bias = None
                if "shortcut" not in name:
                    # normal conv ops
                    if module.padding_mode != "zeros":
                        x = F.conv2d(F.pad(x, module._reversed_padding_repeated_twice, mode=module.padding_mode),
                                     ext_weight, ext_bias, module.stride,
                                     _pair(0), module.dilation, module.groups)
                    x = F.conv2d(x, ext_weight, ext_bias, module.stride,
                                 module.padding, module.dilation, module.groups)
                else:
                    # shortcut conv ops
                    if module.padding_mode != "zeros":
                        x_in = F.conv2d(F.pad(x_in, module._reversed_padding_repeated_twice, mode=module.padding_mode),
                                        ext_weight, ext_bias, module.stride,
                                        _pair(0), module.dilation, module.groups)
                    x_in = F.conv2d(x_in, ext_weight, ext_bias, module.stride,
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
                if "shortcut" not in name:
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
                else:
                    x_in = F.batch_norm(
                        x_in,
                        running_mean=module.running_mean,
                        running_var=module.running_var,
                        weight=ext_weight,
                        bias=ext_bias,
                        training=module.training or not module.track_running_stats,
                        momentum=exponential_average_factor,
                        eps=module.eps
                    )
                    # only valid for resnet-like models
                    x += x_in
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
            elif isinstance(module, torch.nn.AvgPool2d):
                x = F.avg_pool2d(x, module.kernel_size, module.stride,
                                 module.padding, module.ceil_mode, module.count_include_pad, module.divisor_override)
            elif isinstance(module, torch.nn.Flatten):
                x = x.flatten(module.start_dim, module.end_dim)
            elif isinstance(module, torch.nn.Dropout):
                x = F.dropout(x, module.p, module.training, module.inplace)
            elif isinstance(module, torch.nn.Sequential):
                if "shortcut" in name:
                    if len(module) == 0:
                        # perform shortcut connection without conv ops
                        # only valid for resnet-like models
                        x += x_in
                else:
                    # Pass containers
                    pass
            elif isinstance(module, BasicBlock):
                # store the output of the previous block into a buffer
                x_in = x
            else:
                # Warn for other containers
                TypeError("Unexpected {}".format(type(module)))
        return x
