from torchvision import transforms


DEFAULT_INVERSE_TRANSFORM = transforms.Compose([transforms.ToPILImage()])


class TorchDummy:
    """Base class for dummy data

    This module allows easy managing of dummy data

    """

    def __init__(self, _input_shape: list, _label_shape: list, _it=DEFAULT_INVERSE_TRANSFORM):
        self._input_shape = _input_shape
        self._label_shape = _label_shape
        # inverse transform operator
        self._it = _it
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
        self.history.append(self._it(_dummy[0].cpu()))

    def append_label(self, _label):
        self.labels.append(_label)

    def clear_buffer(self):
        """ Clear the history buffer """
        self.history = []
        self.labels = []


class TorchDummyImage(TorchDummy):

    def __init__(self, image_shape: list, n_classes: int, batch_size: int = 1, _it=DEFAULT_INVERSE_TRANSFORM):
        # channel first image for pytorch
        assert len(image_shape) == 3
        # insert the batch dimension
        image_shape.insert(0, batch_size)

        self.n_classes = n_classes
        # label shape [N, C]
        label_shape = [batch_size, self.n_classes]
        super().__init__(_input_shape=image_shape, _label_shape=label_shape, _it=_it)
