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
