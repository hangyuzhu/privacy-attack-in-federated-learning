import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import functools
import os
import PIL
import time
from fleak import model

device = "cuda" if torch.cuda.is_available() else "CPU"

Round = 300
Clinets_per_round = 10
Batch_size = 2048
Gan_epoch = 1

Models = {}
Client_data = {}
Client_labels = {}

BATCH_SIZE = 256

criterion = torch.nn.CrossEntropyLoss().to(device)

class Generator(nn.Module):
    def __init__(self, input_size,num_feature):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.main = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Conv2d(50, 25, 3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.Conv2d(25, 1, 2, stride=2),
            nn.Tanh()
        )
    def forward(self,x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.main(x)
        return x
    # def forward(self, x):
    #     x = self.main(x)
    #     x = x.view(-1, 56 * 56)
    #     return x

# def discriminator_loss(real_output, fake_output, real_labels):
#     real_loss = criterion(real_output,real_labels)
#
#     fake_result = np.zeros(len(fake_output))
#     # Attack label
#     for i in range(len(fake_result)):
#         fake_result[i] = 10
#     fake_loss = criterion(fake_result, fake_output)
#     total_loss = real_loss + fake_loss
#     return total_loss
#
#     # Loss of generator
# def generator_loss(fake_output):
#     ideal_result = np.zeros(len(fake_output))
#     # Attack label
#     for i in range(len(ideal_result)):
#         # The class which attacker intends to get
#         ideal_result[i] = 4
#
#     return criterion(fake_output, ideal_result)

def train(dataset, labels, GAN_epoch, epochs,img_size,generator,discriminator, BATCH_SIZE): ## dataset: 真实用户数据
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    g_optim = torch.optim.Adam(generator.parameters(), lr=0.001)

    for epoch in range(GAN_epoch):
        start_time = time.time()
        for i in range(round(len(dataset) / BATCH_SIZE)):
            image_batch = dataset[i * BATCH_SIZE:min(len(dataset), (i + 1) * BATCH_SIZE)]
            labels_batch = labels[i * BATCH_SIZE:min(len(dataset), (i + 1) * BATCH_SIZE)]
            random_noise = torch.randn(img_size, 100, device=device)

            for j in range(epochs):
                ## training D
                real_output = discriminator(image_batch, training=False)
                dis_loss1 = criterion(real_output, labels_batch)
                fake_image = generator(random_noise)
                fake_output = discriminator(fake_image,training=False)
                dis_loss2 = criterion(fake_output, labels_batch)

                dis_loss = dis_loss1+dis_loss2
                discriminator.zero_grad()
                dis_loss.backward()
                d_optim.step()

                ## training G
                fake_output = discriminator(fake_image).view(-1)
                gen_loss = criterion(fake_output, labels_batch)

                generator.zero_grad()
                gen_loss.backward()
                g_optim.step()

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start_time))

        # Last epoch generate the images and merge them to the dataset
        generate_and_save(generator, epochs, random_noise)


# def train_server(discriminator_loss1, epochs, img_size, generator, discriminator, labels):
#     d_optim = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
#     g_optim = torch.optim.Adam(generator.parameters(), lr=0.001)
#
#     for epoch in range(epochs):
#         start_time = time.time()
#         for i in range(round(len(labels) / BATCH_SIZE)):
#             labels_batch = labels[i * BATCH_SIZE:min(len(labels), (i + 1) * BATCH_SIZE)]
#
#             ## training D
#             discriminator.zero_grad()
#             # real_output = discriminator(image_batch, training=False)
#             # dis_loss1 = criterion(real_output, labels)
#             discriminator_loss1.backward()
#
#             random_noise = torch.randn(img_size, 100, device=device)
#             fake_image = generator(random_noise)
#             fake_output = discriminator(fake_image, training=False)
#             dis_loss2 = criterion(fake_output, labels_batch)
#             dis_loss2.backward()
#             dis_loss = discriminator_loss1+dis_loss2
#             d_optim.step()
#
#             ## training G
#             generator.zero_grad()
#             fake_output = discriminator(fake_image).view(-1)
#             gen_loss = criterion(fake_output, labels_batch)
#             gen_loss.backward()
#             g_optim.step()
#             # real_output = discriminator(image_batch, training=False)
#             # random_noise = torch.randn(img_size, 100, device=device)
#             # generated_image = generator(random_noise, traning=True)
#             #
#             # fake_output = discriminator(generated_image, training=False)
#             # gen_loss = generator_loss(fake_output)
#             # dis_loss = discriminator_loss(real_output, fake_output, real_label=labels_batch)
#             #
#             # d_optim.zero_grad()
#             # dis_loss.backward()
#             # d_optim.step()
#             #
#             # g_optim.zero_grad()
#             # gen_loss.backward()
#             # g_optim.step()
#
#         print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start_time))
#
#         # Last epoch generate the images and merge them to the dataset
#         generate_and_save(generator, epochs, seed)
def generate_and_save(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize
                     =(6, 6))

    for i in range(predictions.shape[0]):
        plt.subplot(6, 6, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

def GAN_attack(discriminator, num_classes,batch_size,img_size,attack_label,dataset,real_labels):
    discriminator.add_module("add_LeakyReLU", nn.LeakyReLU(num_classes, num_classes+1))
    generator = Generator(100 , 3136).to(device)
    random_noise = torch.randn(img_size, 100, device=device)

    train(datasets=dataset, labels=attack_label, GAN_epoch=Gan_epoch, epochs=100, img_size=img_size, generator=generator,discriminator=discriminator,BATCH_SIZE = batch_size)
    prediction = generator(random_noise, training=False)
    gen_image = np.array(prediction)
    gen_label = np.array([1] * len(gen_image))

    return gen_image, gen_label









