import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fleak.data.image_dataset import DatasetSplit
from fleak.model.gan import MnistGenerator, MnistDiscriminator
from fleak.utils.train_eval import train, evaluate

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Constants
BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 150
noise_dim = 100
num_examples_to_generate = 16

# Data
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('../federated_learning/data/mnist', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../federated_learning/data/mnist', train=False, transform=transform)


# Sample to warm up
warmup_dataloader = DataLoader(DatasetSplit(train_dataset, range(3000)), batch_size=BATCH_SIZE, shuffle=True)

generator = MnistGenerator().to(device)
discriminator = MnistDiscriminator().to(device)

warmup_optimizer = optim.Adam(discriminator.parameters())
criterion = nn.CrossEntropyLoss()

for _ in range(20):
    train(discriminator, device, warmup_dataloader, warmup_optimizer, criterion)

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
test_correct = evaluate(discriminator, device, test_loader)
print('\ntest accuracy: ', test_correct / len(test_dataset))

attack_dataloader = DataLoader(DatasetSplit(train_dataset, torch.where(train_dataset.targets == 0)),
                               batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)


#
#
# # Loss of discriminator
# def discriminator_loss(real_output, fake_output, real_labels):
#     real_loss = cross_entropy(real_labels, real_output)
#
#     fake_result = np.zeros(len(fake_output))
#     # Attack label
#     for i in range(len(fake_result)):
#         fake_result[i] = 10
#     fake_loss = cross_entropy(fake_result, fake_output)
#     total_loss = real_loss + fake_loss
#     return total_loss
#
#
# # Loss of generator
# def generator_loss(fake_output):
#     ideal_result = np.zeros(len(fake_output))
#     # Attack label
#     for i in range(len(ideal_result)):
#         ideal_result[i] = 8
#
#     return cross_entropy(ideal_result, fake_output)
#
#
# # Adam optimizer
# generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
#
#
# # Training step
# @tf.function
# def train_step(images, labels):
#     noise = tf.random.normal([BATCH_SIZE, noise_dim])
#
#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#         generated_images = generator(noise, training=True)
#         print(generated_images.shape)
#
#         # real_output is the probability of the mimic number
#         real_output = malicious_discriminator(images, training=False)
#         fake_output = malicious_discriminator(generated_images, training=False)
#
#         gen_loss = generator_loss(fake_output)
#         disc_loss = discriminator_loss(real_output, fake_output, real_labels=labels)
#
#     gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
#     gradients_of_discriminator = disc_tape.gradient(disc_loss, malicious_discriminator.trainable_variables)
#
#     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
#     discriminator_optimizer.apply_gradients(
#         zip(gradients_of_discriminator, malicious_discriminator.trainable_variables))
#
#
# # Train
# def train(dataset, labels, epochs):
#     for epoch in range(epochs):
#         start = time.time()
#         for i in range(round(len(dataset) / BATCH_SIZE)):
#             image_batch = dataset[i * BATCH_SIZE:min(len(dataset), (i + 1) * BATCH_SIZE)]
#             labels_batch = labels[i * BATCH_SIZE:min(len(dataset), (i + 1) * BATCH_SIZE)]
#             train_step(image_batch, labels_batch)
#
#         print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
#
#     # Last epoch generate the images and merge them to the dataset
#     generate_and_save_images(generator, epochs, seed)
#
#
# # Generate images
# def generate_and_save_images(model, epoch, test_input):
#     predictions = model(test_input, training=False)
#     fig = plt.figure(figsize=(4, 4))
#
#     for i in range(predictions.shape[0]):
#         plt.subplot(4, 4, i + 1)
#         plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
#         plt.axis('off')
#
#     plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
#
#
# attack_images = train_images[train_labels == 0]
# attack_labels = train_labels[train_labels == 0]
#
# # Train model
# train(attack_images, attack_labels, EPOCHS)