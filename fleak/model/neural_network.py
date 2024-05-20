""" neural network models constructed by torch.nn.Module

Caution: If the model is required to be wrapped by MetaModel, all the modules should be
         built 'sequentially'. And for those modules not containing trainable parameters,
         try to use nn.Module other than nn.functional method !

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.image_dataset import IMAGE_SHAPE


class MnistLeNet5(nn.Module):

    def __init__(self, num_classes):
        super(MnistLeNet5, self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # conv2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # linear
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # conv1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # conv2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # linear
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MnistConvNet(nn.Module):

    def __init__(self, num_classes):
        super(MnistConvNet, self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.relu1 = nn.ReLU()
        # conv2
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        # linear
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9216, 128)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # conv1
        x = self.conv1(x)
        x = self.relu1(x)
        # conv2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2d(x, 2)
        x = self.dropout1(x)
        # linear
        x = self.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        return self.fc2(x)


class MnistMLP(nn.Module):

    def __init__(self, num_classes):
        super(MnistMLP, self).__init__()
        self.flatten = nn.Flatten()
        # linear1
        self.fc1 = nn.Linear(28 * 28, 200)
        self.relu1 = nn.ReLU()
        # linear2
        self.fc2 = nn.Linear(200, 200)
        self.relu2 = nn.ReLU()
        # linear3
        self.fc3 = nn.Linear(200, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class CifarMLP(nn.Module):

    def __init__(self, num_classes):
        super(CifarMLP, self).__init__()
        self.flatten = nn.Flatten()
        # linear1
        self.fc1 = nn.Linear(3 * 32 * 32, 200)
        self.relu1 = nn.ReLU()
        # linear2
        self.fc2 = nn.Linear(200, 200)
        self.relu2 = nn.ReLU()
        # linear3
        self.fc3 = nn.Linear(200, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class FC2(nn.Module):
    """

    Linear model adopted in CPA

    """

    def __init__(self, num_classes, x_dim=math.prod(IMAGE_SHAPE["tiny_imagenet"]), h_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
        )
        self.final = nn.Linear(h_dim, num_classes)
        self.attack_index = 0
        self.pretrained = False

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.net(x)
        return self.final(x)


class CifarConvNet(nn.Module):
    """ Replacing .view() by nn.Flatten module for MetaModel """

    def __init__(self, num_classes):
        super(CifarConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.drop_out1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*16*16, 512)
        self.relu3 = nn.ReLU()
        self.drop_out2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool(self.relu2(self.conv2(x)))
        x = self.drop_out1(x)
        x = self.flatten(x)
        x = self.drop_out2(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
