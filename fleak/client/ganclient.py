import copy

import torch
import torch.optim as optim
import torch.nn as nn
from ..utils.train_eval import train, evaluate

from .client import Client
from fleak.attack import GAN
from ..model import MnistGenerator

# device = "cuda" if torch.cuda.is_available() else "CPU"


class GanClient(Client):
    def __init__(self,
                 client_id=None,
                 client_group=None,
                 client_model=None,
                 dataset=None,
                 noise_dim=100,
                 num_epochs=1,
                 gan_epochs=1,
                 lr=0.1,
                 lr_decay=0.95,
                 momentum=0.5,
                 train_loader=None,
                 valid_loader=None,
                 test_loader=None,
                 device=None):
        super(GanClient, self).__init__(
            client_id=client_id,
            client_group=client_group,
            client_model=client_model,
            num_epochs=num_epochs,
            lr=lr,
            lr_decay=lr_decay,
            momentum=momentum,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            device=device
        )
        self.discriminator = copy.deepcopy(self.client_model)
        self.generator = MnistGenerator().to(self.device)
        self.D_optimizer = optim.SGD(self.discriminator.parameters(), lr=1e-4, weight_decay=1e-7)
        self.G_optimizer = optim.SGD(self.generator.parameters(), lr=1e-3, weight_decay=1e-7)
        self.noise_dim = noise_dim
        self.fixed_noise = torch.randn(16, self.noise_dim, device=device)

        self.gan_epochs = gan_epochs
        self.dataset = dataset
        # self.img_size = img_size

    def synchronize(self, cur_round, model_params):
        self.cur_round = cur_round
        # inner deep copied
        self.client_model.load_state_dict(model_params)
        # update discriminator
        self.discriminator.load_state_dict(model_params)

    def train(self, verbose=True):
        # gan attack
        for _ in range(self.gan_epochs):
            GAN.attack(discriminator=self.discriminator, generator=self.generator, device=self.device,
                       dataloader=self.train_loader, D_optimizer=self.D_optimizer, G_optimizer=self.G_optimizer,
                       criterion=self.criterion, tracked_class=3, noise_dim=self.noise_dim)
        if verbose:
            GAN.plot_image(self.generator, self.fixed_noise, self.cur_round)
        # local training
        for local_epoch in range(self.num_epochs):
            # local batch training
            train(model=self.client_model,
                  device=self.device,
                  train_loader=self.train_loader,
                  optimizer=self.optimizer,
                  criterion=self.criterion)
        self.optimizer.param_groups[0]["lr"] *= self.lr_decay
        return self.client_id, len(self.train_loader.dataset), self.client_model.state_dict()

    # def adversarial(self, ):

    def evaluate(self, set_to_use='test'):
        assert set_to_use in ['train', 'test', 'valid']
        # return test accuracy of this client
        self.client_model.eval()
        if set_to_use == 'train':
            loader = self.train_loader
        elif set_to_use == 'test':
            loader = self.test_loader
        else:  # valid
            loader = self.valid_loader
        correct = evaluate(model=self.client_model, device=self.device, eval_loader=loader)
        return correct, len(loader.dataset)

    def save(self, path):
        torch.save(self.client_model.state_dict(), path)



