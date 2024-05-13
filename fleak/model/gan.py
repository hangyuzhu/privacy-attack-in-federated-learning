import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistGenerator(nn.Module):

    def __init__(self):
        super(MnistGenerator, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(100, 7 * 7 * 256),   # 256x7x7
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


class MnistDiscriminator(nn.Module):

    def __init__(self):
        super(MnistDiscriminator, self).__init__()
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
        self.fc = nn.Linear(128 * 7 * 7, 11)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc(x)
        return x


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


class Cifar10Generator(nn.Module):
    def __init__(self):
        super(Cifar10Generator, self).__init__()
        self.linear = nn.Sequential(
                    nn.Linear(100, 4 * 4 * 256),
                    nn.BatchNorm1d(4 * 4 * 256),
                    nn.ReLU(True)
                )
                # self.main = nn.Sequential(
                #     nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
                #     nn.BatchNorm2d(64 * 8),
                #     nn.ReLU(True),
        self.main = nn.Sequential(
                    # state size. 256 x 4 x 4
                    nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    # state size. 128 x 8 x 8
                    nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    # state size. 64 x 16 x 16
                    nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                    # nn.Tanh()
                    # nn.BatchNorm2d(64),
                    # nn.ReLU(True),
                    # # state size. 3 x 32 x 32
                    # nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                    nn.Tanh()
                    # state size. (nc) x 64 x 64
                )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 256, 4, 4)
        x = self.main(x)
        return x

class Cifar10Discriminator(nn.Module):
    def __init__(self):
        super(Cifar10Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 64 * 32 * 32
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 128 * 16 * 16
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 256 * 8 * 8
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )


        # self.main = nn.Sequential(
        #     nn.Conv2d(3, 64, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf) x 32 x 32
        #     nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(64 * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*2) x 16 x 16
        #     nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(64 * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*4) x 8 x 8
        #     nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(64 * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*8) x 4 x 4
        #     nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
        #     nn.Sigmoid()
        # )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.main(x)
        x = x.view(-1, 1).squeeze(1)
        return x


class Generator(nn.Module):
    def __init__(self, channel=3, z_hidden=100, g_hidden=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input layer
            nn.ConvTranspose2d(z_hidden, g_hidden * 8, 4, 1, 0, bias=False),  # (g_hidden*8)x4x4
            nn.BatchNorm2d(g_hidden * 8),
            nn.ReLU(True),
            # 1st hidden layer
            nn.ConvTranspose2d(g_hidden * 8, g_hidden * 4, 4, 2, 1, bias=False),  # (g_hidden*4)x8x8
            nn.BatchNorm2d(g_hidden * 4),
            nn.ReLU(True),
            # 2nd hidden layer
            nn.ConvTranspose2d(g_hidden * 4, g_hidden * 2, 4, 2, 1, bias=False),  # (g_hidden*2)x16x16
            nn.BatchNorm2d(g_hidden * 2),
            nn.ReLU(True),
            # 3rd hidden layer
            nn.ConvTranspose2d(g_hidden * 2, g_hidden, 4, 2, 1, bias=False),  # g_hidden x32x32
            nn.BatchNorm2d(g_hidden),
            nn.ReLU(True),
            # output layer
            nn.ConvTranspose2d(g_hidden, channel, 4, 2, 1, bias=False),  # Cx64x64
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, channel=3, d_hidden=64):
        super(Discriminator, self).__init__()
        self.d_hidden = d_hidden

        self.main = nn.Sequential(
            # 1st layer
            nn.Conv2d(channel, d_hidden, 4, 2, 1, bias=False),  # d_hiddenx32x32
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            nn.Conv2d(d_hidden, d_hidden * 2, 4, 2, 1, bias=False),  # (d_hidden*2)x16x16
            nn.BatchNorm2d(d_hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            nn.Conv2d(d_hidden * 2, d_hidden * 4, 4, 2, 1, bias=False),  # (d_hidden*4)x8x8
            nn.BatchNorm2d(d_hidden * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            nn.Conv2d(d_hidden * 4, d_hidden * 8, 4, 2, 1, bias=False),  # (d_hidden*8)x4x4
            nn.BatchNorm2d(d_hidden * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            # nn.Conv2d(d_hidden * 8, 11, 4, 1, 0, bias=False),  # 1x1x1
            # nn.Sigmoid()
        )
        self.fc = nn.Linear(d_hidden * 8 * 4 * 4, 11)

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, self.d_hidden * 8 * 4 * 4)
        return self.fc(x)
