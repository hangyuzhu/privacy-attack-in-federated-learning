import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

"""Train generator of GGL

Official implementation of GGL for training generator on CeleA dataset (32x32)
Redundant part is removed from the original code
We instead train generator of GGL on cifar10 dataset

"""
import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from fleak.model import GGLGenerator
from fleak.model import GGLDiscriminator
from fleak.data.image_dataset import IMAGE_MEAN_GAN, IMAGE_STD_GAN
from fleak.data.image_dataset import UnNormalize


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)

    fake = torch.ones(real_samples.shape[0], 1).to(device)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def main(args):
    lambda_gp = 10   # Loss weight for gradient penalty

    # ----------
    #  Data
    # ----------
    dm = IMAGE_MEAN_GAN["cifar10"]
    ds = IMAGE_STD_GAN["cifar10"]
    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            args.data_path,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(dm, ds)
            ])
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    unnormalize = UnNormalize(IMAGE_MEAN_GAN["cifar10"], IMAGE_STD_GAN["cifar10"])

    # ----------
    #  Model
    # ----------
    generator = GGLGenerator(args.latent_dim).to(args.device)
    discriminator = GGLDiscriminator(args.latent_dim).to(args.device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # ----------
    #  Training
    # ----------

    for _ in tqdm.tqdm(range(args.n_epochs)):
        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = imgs.to(args.device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = torch.randn(imgs.shape[0], args.latent_dim).to(args.device)

            # Generate a batch of images
            fake_imgs = generator(z)

            # Real images
            real_validity = discriminator(real_imgs)
            # Fake images
            fake_validity = discriminator(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, args.device)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % args.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------
                # Generate a batch of images
                fake_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

    z = torch.randn(64, args.latent_dim).to(args.device)
    fake_imgs = generator(z)

    imgs = unnormalize(fake_imgs)
    imgs = make_grid(imgs)

    plt.figure(figsize=(14, 14))
    plt.imshow(np.transpose(imgs.detach().cpu().numpy(), (1, 2, 0)))
    plt.show()

    torch.save(generator.state_dict(), args.save_path)


if __name__ == "__main__":
    import sys
    import os
    import argparse

    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='../../federated_learning/data/cifar10',
                        type=str, help='path of the dataset')
    parser.add_argument('--save_path', default='../saved_models/ggl_cifar10.pth',
                        type=str, help='path of the saved model')

    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")

    parser.add_argument('--device', default='cuda', help='device')

    args = parser.parse_args()
    print(args)

    main(args)
