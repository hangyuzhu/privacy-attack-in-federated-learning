import os
import torch
import matplotlib.pyplot as plt

device="cuda" if torch.cuda.is_available() else "CPU"

def attack(discriminator, generator, device, dataloader, D_optimizer, G_optimizer, criterion,
           tracked_class, noise_dim=100):
    for i, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)
        noise = torch.randn(len(labels), noise_dim, device=device)

        # -----------------
        #  Train Generator
        # -----------------
        generator.train()
        discriminator.eval()
        G_optimizer.zero_grad()

        # Generate a batch of images
        fake_features = generator(noise)
        # Generate the tracked labels
        tracked_labels = torch.full(labels.shape, tracked_class, device=device)

        # Loss measures generator's ability to fool the discriminator
        g_loss = criterion(discriminator(fake_features), tracked_labels)
        g_loss.backward()
        G_optimizer.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        generator.eval()
        discriminator.train()
        D_optimizer.zero_grad()

        fake_features = generator(noise)
        # Generate fake labels
        fake_labels = torch.full(labels.shape, 10, device=device)

        # Measure discriminator's ability to classify real from generated samples
        real_loss = criterion(discriminator(features), labels)
        fake_loss = criterion(discriminator(fake_features.detach()), fake_labels)
        d_loss = real_loss + fake_loss

        d_loss.backward()
        D_optimizer.step()


def generate_save_images(generator, fixed_noise, path, data_name):
    dm = torch.as_tensor([0.4914, 0.4822, 0.4465], device=device, dtype=torch.float32)[:, None, None]
    ds = torch.as_tensor([0.2023, 0.1994, 0.2010], device=device, dtype=torch.float32)[:, None, None]
    generator.eval()
    with torch.no_grad():
        # fake = generator(fixed_noise).detach().cpu()
        fake = generator(fixed_noise).detach()

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        # ndarr = fake[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        ndarr = fake[i].mul_(ds).add_(dm).clamp_(min=0, max=1).permute(1, 2, 0).to("cpu", torch.float32).numpy()
        # plt.imshow(ndarr, cmap='gray')
        plt.imshow(ndarr)
        plt.axis('off')
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, data_name + '_fake_image.png'))
