"""Deep Models Under the GAN: Information Leakage from Collaborative Deep Learning

https://dl.acm.org/doi/10.1145/3133956.3134012

"""

import os
import time
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


def dmgan(tracked_class,
          generator,
          discriminator,
          dataloader,
          dummy,
          rec_epochs,
          d_optimizer,
          g_optimizer,
          criterion,
          noise_dim,
          device) -> None:
    """Client side attack

    The structure of discriminator (global model) should be modified
    Be careful about train& eval mode of both generator and discriminator

    :param tracked_class: tracked label class
    :param generator: local generator
    :param discriminator: equivalent to the global model
    :param dataloader: local dataloader
    :param dummy: TorchDummy object
    :param rec_epochs: the number of reconstruct epochs
    :param d_optimizer: optimizer of the discriminator
    :param g_optimizer: optimizer of the generator
    :param criterion: loss criterion
    :param noise_dim: dimension of the random noise, default 100
    :param device: cpu or gpu
    :return: None
    """
    fixed_noise = torch.randn(dummy.batch_size, 100, device=device)

    pbar = tqdm(range(rec_epochs),
                total=rec_epochs,
                desc=f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}')
    for _ in pbar:
        for i, (features, labels) in enumerate(dataloader):
            features, labels = features.to(device), labels.to(device)
            noise = torch.randn(len(labels), noise_dim, device=device)

            # -----------------
            #  Train Generator
            # -----------------
            generator.train()
            discriminator.eval()
            g_optimizer.zero_grad()

            # Generate a batch of images
            fake_features = generator(noise)
            # Generate the tracked labels
            tracked_labels = torch.full(labels.shape, tracked_class, device=device)

            # Loss measures generator's ability to fool the discriminator
            g_loss = criterion(discriminator(fake_features), tracked_labels)
            g_loss.backward()
            g_optimizer.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            generator.eval()
            discriminator.train()
            d_optimizer.zero_grad()

            fake_features = generator(noise)
            # Generate fake labels
            fake_labels = torch.full(labels.shape, 10, device=device)

            # Measure discriminator's ability to classify real from generated samples
            real_loss = criterion(discriminator(features), labels)
            fake_loss = criterion(discriminator(fake_features.detach()), fake_labels)
            d_loss = real_loss + fake_loss

            d_loss.backward()
            d_optimizer.step()

            pbar.set_description("Real Loss {:.6}, Fake Loss {:.6}".format(real_loss, fake_loss))

    generator.eval()
    with torch.no_grad():
        dummy_data = generator(fixed_noise)
        dummy.append(dummy_data)


def save_generated_images(generator, fixed_noise, path, data_name):
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        ndarr = fake[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        plt.imshow(ndarr, cmap='gray')
        plt.axis('off')
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, data_name + '_fake_image.png'))
