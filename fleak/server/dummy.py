import torch
import numpy as np
from torchvision import transforms


DEFAULT_INVERSE_TRANSFORM = transforms.Compose([transforms.ToPILImage()])


class TorchDummy:
    """Base class for dummy data

    This module allows easy managing of dummy data

    """

    def __init__(self, dataset, batch_size=1, _it=DEFAULT_INVERSE_TRANSFORM):
        """
        :param dataset: torch Dataset
        :param batch_size: batch size of dummy data
        :param _it: inverse transform
        """
        _data_shape = list(dataset.data.shape)
        # convert to channel first
        _data_shape[1], _data_shape[-1] = _data_shape[-1], _data_shape[1]
        # change the batch size
        _data_shape[0] = batch_size
        self._data_shape = _data_shape

        self.n_classes: int = len(set(np.array(dataset.targets)))
        self._label_shape = [batch_size, self.n_classes]

        self._it = _it

        # buffer
        self.history = []

    @property
    def data_shape(self):
        return self._data_shape

    @property
    def label_shape(self):
        return self._label_shape

    def append(self, _dummy):
        if isinstance(_dummy, list):
            self.history.append([self._it(_dummy[0].detach().cpu()), _dummy[1]])
        elif isinstance(_dummy, torch.Tensor):
            self.history.append(self._it(_dummy.detach().cpu()))
        else:
            raise TypeError("{} is not an expected data type".format(type(_dummy)))

    def clear_buffer(self):
        """ Clear the history buffer """
        self.history = []
