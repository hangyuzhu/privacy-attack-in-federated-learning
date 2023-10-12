import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils

vutils

from fleak.data.image_dataset import DatasetSplit
from fleak.model.gan_network import MnistGenerator, MnistDiscriminator, Generator, Discriminator
from fleak.utils.train_eval import train, evaluate


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Constants
BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 150
noise_dim = 100

# Data
transform = transforms.Compose([
        # transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('../federated_learning/data/mnist', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../federated_learning/data/mnist', train=False, transform=transform)

# fixed noise
fixed_noise = torch.randn(16, noise_dim, device=device)

# Sample to warm up
warmup_dataloader = DataLoader(DatasetSplit(train_dataset, range(3000)), batch_size=BATCH_SIZE, shuffle=True)

generator = MnistGenerator().to(device)
discriminator = MnistDiscriminator().to(device)
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

warmup_optimizer = optim.Adam(discriminator.parameters())
criterion = nn.CrossEntropyLoss()

for _ in range(20):
    train(discriminator, device, warmup_dataloader, warmup_optimizer, criterion)

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
test_correct = evaluate(discriminator, device, test_loader)
print('\ntest accuracy: ', test_correct / len(test_dataset))

attack_dataloader = DataLoader(DatasetSplit(train_dataset, torch.where(train_dataset.targets == 8)[0]),
                               batch_size=BATCH_SIZE, shuffle=True)
# attack_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# optimizer
optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)

for epoch in range(EPOCHS):
    for i, (features, labels) in enumerate(attack_dataloader):
        features, labels = features.to(device), labels.to(device)
        noise = torch.randn(len(labels), noise_dim, device=device)

        # -----------------
        #  Train Generator
        # -----------------
        generator.train()
        discriminator.eval()
        optimizer_G.zero_grad()

        # Generate a batch of images
        fake_features = generator(noise)
        # Generate the tracked labels
        tracked_labels = torch.full(labels.shape, 8, device=device)

        # Loss measures generator's ability to fool the discriminator
        g_loss = criterion(discriminator(fake_features), tracked_labels)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        generator.eval()
        discriminator.train()
        optimizer_D.zero_grad()

        fake_features = generator(noise)
        # Generate fake labels
        fake_labels = torch.full(labels.shape, 10, device=device)

        # Measure discriminator's ability to classify real from generated samples
        real_loss = criterion(discriminator(features), labels)
        fake_loss = criterion(discriminator(fake_features.detach()), fake_labels)
        d_loss = real_loss + fake_loss

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, EPOCHS, i, len(attack_dataloader), d_loss.item(), g_loss.item())
        )

        # Generate fake images
        batches_done = epoch * len(attack_dataloader) + i
        if batches_done % 400 == 0:
            generator.eval()
            with torch.no_grad():
                predictions = generator(fixed_noise).detach().cpu()
            # f_imgs = vutils.make_grid(predictions, padding=2, normalize=True)
            # plt.axis('off')
            # plt.imshow(np.transpose(f_imgs, (1, 2, 0)))
            # plt.show()

            for i in range(9):
                plt.subplot(3, 3, i + 1)
                ndarr = predictions[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                plt.imshow(ndarr, cmap='gray')
                plt.axis('off')
            plt.show()


# for epoch in range(EPOCHS):
#     for i, (features, labels) in enumerate(attack_dataloader):
#         # ---------------------
#         #  Train Discriminator
#         # ---------------------
#         generator.eval()
#         discriminator.train()
#         optimizer_D.zero_grad()
#
#         # Update D with real data
#         features, labels = features.to(device), labels.to(device)
#         real_loss = criterion(discriminator(features), labels)
#         real_loss.backward()
#
#         # Update D with fake data
#         noise = torch.randn(len(labels), noise_dim, device=device)
#         fake_features = generator(noise)
#         fake_labels = torch.full(labels.shape, 10, device=device)
#         fake_loss = criterion(discriminator(fake_features.detach()), fake_labels)
#         fake_loss.backward()
#
#         d_loss = real_loss + fake_loss
#         optimizer_D.step()
#
#         # -----------------
#         #  Train Generator
#         # -----------------
#         generator.train()
#         discriminator.eval()
#         optimizer_G.zero_grad()
#
#         fake_features = generator(noise)
#         tracked_labels = torch.full(labels.shape, 8, device=device)
#
#         # Loss measures generator's ability to fool the discriminator
#         g_loss = criterion(discriminator(fake_features), tracked_labels)
#         g_loss.backward()
#         optimizer_G.step()
#
#         print(
#             "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#             % (epoch, EPOCHS, i, len(attack_dataloader), d_loss.item(), g_loss.item())
#         )
#
#         # Generate fake images
#         batches_done = epoch * len(attack_dataloader) + i
#         if batches_done % 400 == 0:
#             generator.eval()
#             with torch.no_grad():
#                 predictions = generator(fixed_noise).detach().cpu()
#             f_imgs = vutils.make_grid(predictions, padding=2, normalize=True)
#             plt.axis('off')
#             plt.imshow(np.transpose(f_imgs, (1, 2, 0)))
#             plt.show()
#
#             # for i in range(predictions.shape[0]):
#             #     plt.subplot(3, 3, i + 1)
#             #     ndarr = predictions[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
#             #     plt.imshow(ndarr, cmap='gray')
#             #     plt.axis('off')
#             # plt.show()
