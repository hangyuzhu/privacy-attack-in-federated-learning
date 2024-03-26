import torch.nn as nn


class RobinFed(nn.Module):

    def __init__(self, ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear0 = nn.Linear(
        self.linear1 = nn.Linear()
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=tuple(shape_img[1:]))
        self.base =

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(self.linear0(x))
        x = self.unflatten(x)
        x = self.base(x)
        return x

