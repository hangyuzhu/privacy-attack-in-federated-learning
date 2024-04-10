import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from fleak.model.gan_network import Cifar10Generator,Cifar10Discriminator

device="cuda" if torch.cuda.is_available() else "CPU"


def gan_train(discriminator, generator, dataloader,D_optimizer, G_optimizer, criterion,noise_dim=100):
    real_label = 1
    fake_label = 0
    for i, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)
        noise = torch.randn(len(labels), noise_dim, device=device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        generator.eval()
        discriminator.train()
        D_optimizer.zero_grad()
        # train with real
        real_data = features.to(device)
        label = torch.full(labels.shape, real_label, device=device)

        output = discriminator(real_data)
        real_loss = criterion(output.float(), label.float())
        # train with fake
        fake_data = generator(noise)
        label.fill_(fake_label)
        output = discriminator(fake_data.detach())
        fake_loss = criterion(output.float(), label.float())

        d_loss = real_loss + fake_loss
        d_loss.backward()
        D_optimizer.step()

        # -----------------
        #  Train Generator
        # -----------------
        generator.train()
        discriminator.eval()
        G_optimizer.zero_grad()

        # Generate a batch of images
        fake_features = generator(noise)
        # Generate the tracked labels
        label.fill_(real_label)

        # Loss measures generator's ability to fool the discriminator
        g_loss = criterion(discriminator(fake_features).float(), label.float())
        g_loss.backward()
        G_optimizer.step()


    torch.save(generator.state_dict(), 'models_parameter/cifarGGLgenerator.pth')


if __name__ == '__main__':
    generator = Cifar10Generator()
    generator = generator.to(device)
    discriminator = Cifar10Discriminator()
    discriminator = discriminator.to(device)
    D_optimizer = optim.SGD(discriminator.parameters(), lr=1e-3, weight_decay=1e-7)
    G_optimizer = optim.SGD(generator.parameters(), lr=1e-4, weight_decay=1e-7)
    criterion = nn.CrossEntropyLoss().to(device)
    noise_dim = 100
    dataset = torchvision.datasets.CIFAR10(root="C:/Users/merlin/data", download=True,
                            transform=transforms.Compose([
                                # transforms.Resize(64),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                             shuffle=True, num_workers=2)
    gan_train(discriminator, generator, dataloader, D_optimizer, G_optimizer, criterion, noise_dim)