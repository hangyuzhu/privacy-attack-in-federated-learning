import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistGenerator(nn.Module):

    def __init__(self, nz=100):
        super(MnistGenerator, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(nz, 7 * 7 * 256),   # 256x7x7
            nn.BatchNorm1d(7 * 7 * 256),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, 1, 2, bias=False),  # 128x7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 64x14x14
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),  # 1x28x28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 256, 7, 7)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class DMGanMnistDiscriminator(nn.Module):

    def __init__(self, num_classes):
        super(DMGanMnistDiscriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),   # 64x14x14
            nn.LeakyReLU(),
            nn.Dropout2d(0.3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 128x7x7
            nn.LeakyReLU(),
            nn.Dropout2d(0.3)
        )
        # add an extra classification class in the output layer
        self.fc = nn.Linear(128 * 7 * 7, num_classes + 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc(x)
        return x


class CifarGenerator(nn.Module):

    def __init__(self, nz=100):
        super(CifarGenerator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, nz x 1 x 1
            nn.ConvTranspose2d(nz, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 128 x 4 x 4
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 8 x 8
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 16 x 16
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. 3 x 32 x 32
        )

    def forward(self, input):
        input = input.view(-1, input.size(1), 1, 1)
        output = self.main(input)
        return output


class CifarDiscriminator(nn.Module):

    def __init__(self):
        super(CifarDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # input is 3 x 32 x 32
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 16 x 16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 8 x 8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 4 x 4
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size. 1 x 1 x 1
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class DMGanCifarDiscriminator(nn.Module):

    def __init__(self, num_classes):
        super(DMGanCifarDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # input is 3 x 32 x 32
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 16 x 16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 8 x 8
            nn.Flatten(),
            # add an extra classification class in the output layer
            nn.Linear(128 * 8 * 8, num_classes + 1)
        )

    def forward(self, input):
        return self.main(input)


class GLU(nn.Module):
    """ Gated Linear Unit

    Language Modeling with Gated Convolutional Networks
    http://proceedings.mlr.press/v70/dauphin17a/dauphin17a.pdf

    This module can be regarded as an activation layer
    The authors believe that GLU is far more stable than ReLU and can learn faster than Sigmoid

    """

    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        # input channels are divided by 2
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class GRNNGenerator(nn.Module):

    def __init__(self, num_classes, in_features, image_shape):
        """ Generator for GRNN

        :param num_classes: number of classification classes
        :param in_features: dimension of the input (noise) features
        :param image_shape: channel first image shape
        """
        super(GRNNGenerator, self).__init__()
        # dummy label predictions
        self.linear = nn.Linear(in_features, num_classes)

        # for dummy data
        image_channel = image_shape[0]
        image_size = image_shape[1]
        block_nums = int(math.log2(image_size) - 3)
        # (B, 128, 1, 1) -> (B, 128, 4, 4)
        self.in_block = nn.Sequential(
            # channels times 2
            nn.ConvTranspose2d(in_features, image_size * pow(2, block_nums) * 2, 4, 1, 0),
            GLU()   # channels are divided by 2
        )
        self.blocks = nn.ModuleList()
        # (B, 128, 4, 4) -> (B, 64, 8, 8) -> (B, 32, 16, 16)
        for bn in reversed(range(block_nums)):
            self.blocks.append(self.up_sampling(pow(2, bn + 1) * image_size, pow(2, bn) * image_size))
        # (B, 32, 16, 16) -> (B, 3, 32, 32)
        self.out_block = self.up_sampling(image_size, image_channel)

    @staticmethod
    def up_sampling(in_planes, out_planes):
        return nn.Sequential(
            # image rows and cols doubled
            nn.Upsample(scale_factor=2, mode='nearest'),
            # padding makes the image size unchanged
            nn.Conv2d(in_planes, out_planes * 2, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            GLU()
        )

    # forward method
    def forward(self, x):
        # generate dummy label
        y = F.softmax(self.linear(x), -1)

        # generate dummy data
        # (B, In) -> (B, In, 1, 1)
        x = x.view(-1, x.size(1), 1, 1)
        x = self.in_block(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_block(x)
        x = F.sigmoid(x)
        return x, y


def _init_normal(m, mean, std):
    """ initialize the model parameters by random variables sampled from Gaussian distribution

    Caution: 1) this should be correctly employed by 'model.apply()'
             2) may be unnecessary for GRNN if the discriminator is not initialized by Gaussian

    :param m: nn.Module
    :return: None
    """
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean, std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class GGLGenerator(nn.Module):
    """Generator of GGL

    This is the official implementation for CeleA dataset (resized to 32x32)
    https://github.com/zhuohangli/GGL/blob/main/GAN-training/wgan-gp_celeba.ipynb

    """

    def __init__(self, dim=128):
        super(GGLGenerator, self).__init__()
        self.dim = dim

        self.linear = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * dim),
            nn.BatchNorm1d(4 * 4 * 4 * dim),
            nn.ReLU(True),
        )
        # (B, 4 * dim * 4 * 4)
        self.main = nn.Sequential(
            # (B, 4 * dim, 4, 4)
            nn.ConvTranspose2d(4 * dim, 2 * dim, 2, stride=2),
            nn.BatchNorm2d(2 * dim),
            nn.ReLU(True),
            # (B, 2 * dim, 8, 8)
            nn.ConvTranspose2d(2 * dim, dim, 2, stride=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            # (B, dim, 16, 16)
            nn.ConvTranspose2d(dim, 3, 2, stride=2),
            nn.Tanh()
            # (B, 3, 32, 32)
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 4 * self.dim, 4, 4)
        x = self.main(x)
        return x


class GGLDiscriminator(nn.Module):
    """Discriminator of GGL

    Official implementation released by authors of the paper
    https://github.com/zhuohangli/GGL/blob/main/GAN-training/wgan-gp_celeba.ipynb

    """

    def __init__(self, dim=128):
        super(GGLDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # (B, 3, 32, 32)
            nn.Conv2d(3, dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            # (B, dim, 16, 16)
            nn.Conv2d(dim, 2 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            # (B, 2 * dim, 8, 8)
            nn.Conv2d(2 * dim, 4 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            # (B, 4* dim, 4, 4)
            nn.Flatten(),
            nn.Linear(4 * 4 * 4 * dim, 1)
        )

    def forward(self, x):
        return self.main(x)