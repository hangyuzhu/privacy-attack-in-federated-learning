import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

from fleak.model.neural_network import CifarConvNet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
# torch.manual_seed(50)


dst = datasets.CIFAR10("../federated_learning/data/cifar10", download=True)
# tp = transforms.Compose([
#     transforms.Resize(32),
#     transforms.CenterCrop(32),
#     transforms.ToTensor()
# ])
# dm = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
# ds = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

dm = (0.4914, 0.4822, 0.4465)
ds = (0.2023, 0.1994, 0.2010)

# dm = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
# ds = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]


tp = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dm, ds)
    ])


class UnNormalize(torchvision.transforms.Normalize):
    def __init__(self, mean, std, *args, **kwargs):
        new_mean = [-m/s for m, s in zip(mean, std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)

# tt = transforms.ToPILImage()
tt = transforms.Compose([
    # transforms.ToPILImage(),
    UnNormalize(dm, ds),
    transforms.ToPILImage()
])

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

# DONOT fully understand but seems like putting the label into the tensor in the available device
def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

# Note the use of log-softmax and not just log of predictions normalized
def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


# Very cool way to initializing all the weights
# How does one initialize weights with different distribution in different layers
def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         act = nn.Sigmoid
#         self.body = nn.Sequential(
#             nn.Conv2d(3, 12, kernel_size=5, padding=5 // 2, stride=2),
#             act(),
#             nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
#             act(),
#             nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
#             act(),
#             nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
#             act(),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(768, 100)
#         )
#
#     def forward(self, x):
#         out = self.body(x)
#         out = out.view(out.size(0), -1)
#         # print(out.size())
#         out = self.fc(out)
#         return out


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 10)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out


# net = LeNet().to(device)
#
# net.apply(weights_init)

net = CifarConvNet(10).to(device)

criterion = cross_entropy_for_onehot

######### honest partipant #########
img_index = 333
gt_data = tp(dst[img_index][0]).to(device)
gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
gt_label = gt_label.view(1, )

# gt_onehot_label = label_to_onehot(gt_label, num_classes=100)
gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

plt.imshow(tt(gt_data[0].cpu()))
plt.show()
plt.title("Ground truth image")
print("GT label is %d." % gt_label.item(), "\nOnehot label is %d." % torch.argmax(gt_onehot_label, dim=-1).item())

# compute original gradient
out = net(gt_data)
y = criterion(out, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters())


# share the gradients with other clients
original_dy_dx = list((_.detach().clone() for _ in dy_dx))

# generate dummy data and label
# dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
dummy_data = torch.empty(gt_data.size()).to(device).requires_grad_(True)
nn.init.kaiming_uniform_(dummy_data, a=math.sqrt(5))
# dummy_label = torch.empty(gt_onehot_label.size()).to(device).requires_grad_(True)
# nn.init.kaiming_uniform_(dummy_label, a=math.sqrt(5))
# dummy_label = gt_onehot_label.clone().to(device).requires_grad_(True)


plt.imshow(tt(dummy_data[0].cpu()))
plt.title("Dummy data")
print("Dummy label is %d." % torch.argmax(dummy_label, dim=-1).item())

optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

history = []
for iters in range(300):
    def closure():
        optimizer.zero_grad()

        pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(pred,
                               dummy_onehot_label)  # TODO: fix the gt_label to dummy_label in both code and slides.
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

        grad_diff = 0
        grad_count = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx):  # TODO: fix the variablas here
            grad_diff += ((gx - gy) ** 2).sum()
            grad_count += gx.nelement()
        # grad_diff = grad_diff / grad_count * 1000
        grad_diff.backward()

        return grad_diff


    optimizer.step(closure)
    if iters % 10 == 0:
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
    history.append(tt(dummy_data[0].cpu()))

plt.figure(figsize=(12, 8))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i * 10])
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')
plt.show()
print("Dummy label is %d." % torch.argmax(dummy_label, dim=-1).item())