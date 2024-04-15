import torch
import numpy as np
from torchvision import transforms


tp = transforms.Compose([transforms.ToPILImage()])


class TorchDummy:
    """Base class for dummy data

    This module allows easy managing of dummy data

    """

    def __init__(self, dataset, batch_size=1):
        """
        :param dataset: torch Dataset
        :param batch_size: batch size of dummy data
        """
        _data_shape = list(dataset.data.shape)
        # convert to channel first
        _data_shape[1], _data_shape[-1] = _data_shape[-1], _data_shape[1]
        # change the batch size
        _data_shape[0] = batch_size
        self._data_shape = _data_shape

        self.n_classes: int = len(set(np.array(dataset.targets)))
        self._label_shape = [batch_size, self.n_classes]
        # buffer
        self.history = []

    @property
    def data_shape(self):
        return self._data_shape

    @property
    def label_shape(self):
        return self._label_shape

    def append(self, dummy_data: torch.Tensor):
        self.history.append(tp(dummy_data.detach().cpu()))

    def clear_buffer(self):
        self.history = []
