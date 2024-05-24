import math
from typing import Union

import torch
import torch.nn as nn
from torchvision import transforms

from ..data.image_dataset import UnNormalize


class TorchDummy:
    """Base class for dummy data

    This module allows easy managing of dummy data
    Caution: 1) dm & ds are not always available
             2) methods like inverting linear layer does not care about the batch size !

    """

    def __init__(
        self,
        _input_shape: list,
        _label_shape: list,
        batch_size: int,
        dm: Union[list, tuple],
        ds: Union[list, tuple],
        device: str
    ):
        assert _input_shape[0] == batch_size
        assert _label_shape[0] == batch_size
        self.device = device
        self._input_shape = _input_shape
        self._label_shape = _label_shape
        self.batch_size = batch_size

        self.dm = dm
        self.ds = ds

        # buffer
        self.history = []
        self.labels = []

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def label_shape(self):
        return self._label_shape

    def append(self, _dummy):
        self.history.append(_dummy)

    def append_label(self, _label):
        self.labels.append(_label)

    def clear_buffer(self):
        """ Clear the history buffer """
        self.history = []
        self.labels = []

    def generate_dummy_input(self, device=None):
        if device is None:
            device = self.device
        return torch.randn(self.input_shape).to(device).requires_grad_(True)

    def generate_dummy_label(self, device=None):
        if device is None:
            device = self.device
        return torch.randn(self.label_shape).to(device).requires_grad_(True)


class TorchDummyImage(TorchDummy):

    def __init__(
        self,
        image_shape: list,
        batch_size: int,
        n_classes: int,
        dm: Union[list, tuple],
        ds: Union[list, tuple],
        device: str
    ):
        """

        Caution: methods like rtf and GGL do not care about the batch size !

        :param image_shape: 3D image shape
        :param batch_size: batch size
        :param n_classes: number of data classes
        :param dm: normalized mean value
        :param ds: normalized std value
        :param device: running device
        """
        # channel first image for pytorch
        assert len(image_shape) == 3
        self._image_shape = image_shape

        self.n_classes = n_classes
        # label shape [N, C]
        label_shape = [batch_size, self.n_classes]
        super().__init__(
                _input_shape=[batch_size, *image_shape],
                _label_shape=label_shape,
                batch_size=batch_size,
                dm=dm,
                ds=ds,
                device=device,
        )
        # inverse transform operator
        self._it = transforms.Compose([
            UnNormalize(dm, ds),
            transforms.ToPILImage()
        ])

        self.t_dm = torch.as_tensor(self.dm, device=device)[:, None, None]
        self.t_ds = torch.as_tensor(self.ds, device=device)[:, None, None]

    @property
    def image_shape(self):
        return self._image_shape

    def append(self, _dummy, method="ds"):
        if method == "ds":
            self.history.extend([self._it(x.detach().cpu()) for x in _dummy])
        elif method == "infer":
            orig_shape = _dummy.shape
            n = orig_shape[0]
            _dummy = _dummy.view([n, -1])
            _dummy = (_dummy - _dummy.min(dim=-1, keepdim=True)[0]) / (
                    _dummy.max(dim=-1, keepdim=True)[0] - _dummy.min(dim=-1, keepdim=True)[0]
            )
            _dummy = _dummy.view(orig_shape)
            for _img in _dummy:
                _img = _img.permute(1, 2, 0).detach().cpu()
                # retrieve image data without the requirement of mean and std
                _img = (_img - _img.min()) / (_img.max() - _img.min())
                self.history.append(_img)
        else:
            raise ValueError(f"Unknown method {method}")

    def append_label(self, _label):
        self.labels.extend([label.item() for label in _label])


def generate_dummy_k(dummy, device):
    """ Generate dummy data with Kaiming initialization

     This may be helpful for stable generation

    :param dummy: TorchDummy object
    :param device: cpu or cuda
    :return: dummy data & dummy label
     """
    dummy_data = torch.empty(dummy.input_shape).to(device).requires_grad_(True)
    # equivalent to the default initialization of pytorch
    nn.init.kaiming_uniform_(dummy_data, a=math.sqrt(5))
    dummy_label = torch.empty(dummy.label_shape).to(device).requires_grad_(True)
    nn.init.kaiming_uniform_(dummy_label, a=math.sqrt(5))
    return dummy_data, dummy_label


def generate_dummy(dummy, device):
    """ Generate dummy data with Gaussian distribution

    :param dummy: TorchDummy object
    :param device: cpu or cuda
    :return: dummy data & dummy label
    """
    dummy_data = torch.randn(dummy.input_shape).to(device).requires_grad_(True)
    dummy_label = torch.randn(dummy.label_shape).to(device).requires_grad_(True)
    return dummy_data, dummy_label
