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
noise_dim = 100
num_examples_to_generate = 36
num_to_merge = 500
# num_to_merge = 50
seed = torch.random.normal([num_examples_to_generate, noise_dim])
seed_merge = torch.random.normal([num_to_merge, noise_dim])
criterion = torch.nn.CrossEntropyLoss().to(device)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 32 * 32)
        return x

def discriminator_loss(real_output, fake_output, real_labels):
    real_loss = criterion(real_labels, real_output)

    fake_result = np.zeros(len(fake_output))
    # Attack label
    for i in range(len(fake_result)):
        fake_result[i] = 10
    fake_loss = criterion(fake_result, fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

    # Loss of generator
def generator_loss(fake_output):
    ideal_result = np.zeros(len(fake_output))
    # Attack label
    for i in range(len(ideal_result)):
        # The class which attacker intends to get
        ideal_result[i] = 4

    return criterion(ideal_result, fake_output)

def train(dataset, labels, epochs, img_size,generator,discriminator): ## dataset: 真实用户数据，但是不知道怎么写在代码里面
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    g_optim = torch.optim.Adam(generator.parameters(), lr=0.001)

    for epoch in range(epochs):
        start_time = time.time()
        for i in range(round(len(dataset) / BATCH_SIZE)):
            image_batch = dataset[i * BATCH_SIZE:min(len(dataset), (i + 1) * BATCH_SIZE)]
            labels_batch = labels[i * BATCH_SIZE:min(len(dataset), (i + 1) * BATCH_SIZE)]
            random_noise = torch.randn(img_size, 100, device=device)

            generated_image = generator(random_noise, traning=True)
            real_output = discriminator(image_batch, training=False)
            fake_output = discriminator(generated_image, training=False)
            gen_loss = generator_loss(fake_output)
            dis_loss = discriminator_loss(real_output, fake_output, real_label=labels_batch)

            d_optim.zero_grad()
            dis_loss.backward()
            d_optim.step()

            g_optim.zero_grad()
            gen_loss.backward()
            g_optim.step()

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start_time))

        # Last epoch generate the images and merge them to the dataset
        generate_and_save(generator, epochs, seed)

def generate_and_save(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(6, 6))

    for i in range(predictions.shape[0]):
        plt.subplot(6, 6, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

def GAN_attack(discriminator, num_classes,batch,img_size):
    discriminator.add_module("add_LeakyReLU", nn.LeakyReLU(num_classes, num_classes+1))
    generator = Generator().to(device)

    ## attack
    if batch == 1:
        train(epochs=Gan_epoch, img_size=img_size)
        prediction = generator(seed_merge, training=False)
        gen_image = np.array(prediction)
        gen_label = np.array([1] * len(gen_image))

    return gen_image, gen_label






