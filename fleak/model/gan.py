import torch
import torch.nn as nn


class MnistGenerator(nn.Module):
    def __init__(self):
        super(MnistGenerator, self).__init__()

        # self.init_size = opt.img_size // 4
        # self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        # self.conv_blocks = nn.Sequential(
        #     nn.BatchNorm2d(128),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(128, 128, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(128, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(128, 64, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(64, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
        #     nn.Tanh(),
        # )
        self.linear1 = nn.Sequential(
            nn.Linear(100, 7 * 7 * 256),
            nn.BatchNorm1d(7 * 7 * 256),
            nn.LeakyReLU()
        )
        self.conv1 = nn.Sequential(
            # B 256 7 7
            nn.ConvTranspose2d(256, 128, 5, 1, 2, bias=False),  # 128 7 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 64 14 14
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),  # 1 28 28
            nn.Tanh()
        )

    def forward(self, x):
        # out = self.l1(z)
        # out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        # img = self.conv_blocks(out)
        x = self.linear1(x)
        x = x.view(-1, 256, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class MnistDiscriminator(nn.Module):
    def __init__(self):
        super(MnistDiscriminator, self).__init__()

        # def discriminator_block(in_filters, out_filters, bn=True):
        #     block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
        #     if bn:
        #         block.append(nn.BatchNorm2d(out_filters, 0.8))
        #     return block
        #
        # self.model = nn.Sequential(
        #     *discriminator_block(opt.channels, 16, bn=False),
        #     *discriminator_block(16, 32),
        #     *discriminator_block(32, 64),
        #     *discriminator_block(64, 128),
        # )
        #
        # # The height and width of downsampled image
        # ds_size = opt.img_size // 2 ** 4
        # self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),   # 64 14 14
            nn.LeakyReLU(),
            nn.Dropout2d(0.3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),  # 128 7 7
            nn.LeakyReLU(),
            nn.Dropout2d(0.3)
        )

        self.linear = nn.Linear(128 * 7 * 7, 11)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 128 * 7 * 7)
        x = self.linear(x)
        return x

    # def forward(self, img):
    #     out = self.model(img)
    #     out = out.view(out.shape[0], -1)
    #     validity = self.adv_layer(out)
    #
    #     return validity

# generator = MnistGenerator()
# x = torch.randn(10, 100)
# y = generator(x)
# print(y.shape)

# x = torch.randn(10, 1, 28, 28)
# m = MnistDiscriminator()
# print(m(x).shape)