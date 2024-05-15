import torch
import torch.optim as optim
from ..utils.train_eval import evaluate

from .client import Client
from fleak.attack import GAN
from ..utils.train_eval import train
from ..model import MnistGenerator


class GanClient(Client):
    def __init__(self,
                 client_id=None,
                 client_group=None,
                 client_model=None,
                 data_name=None,
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
        self.generator = MnistGenerator().to(self.device)
        # self.generator = Cifar10Generator().to(self.device)
        # discriminator optimizer
        self.D_optimizer = optim.SGD(self.client_model.parameters(), lr=1e-3, weight_decay=1e-7)
        # generator optimizer
        self.G_optimizer = optim.SGD(self.generator.parameters(), lr=1e-4, weight_decay=1e-7)
        self.noise_dim = noise_dim
        self.fixed_noise = torch.randn(16, self.noise_dim, device=device)

        self.gan_epochs = gan_epochs
        self.data_name = data_name

    def synchronize(self, cur_round, model_params):
        self.cur_round = cur_round
        # update discriminator
        self.client_model.load_state_dict(model_params)

    def train(self):
        # gan attack
        for _ in range(self.gan_epochs):
            GAN.attack(discriminator=self.client_model, generator=self.generator, device=self.device,
                       dataloader=self.train_loader, D_optimizer=self.D_optimizer, G_optimizer=self.G_optimizer,
                       criterion=self.criterion, tracked_class=2, noise_dim=self.noise_dim)
        # if verbose and (self.cur_round + 1) % 50 == 0:
        GAN.generate_save_images(self.generator, self.fixed_noise, 'saved_results', self.data_name)
        for local_epoch in range(self.num_epochs):
            # local batch training
            train(model=self.client_model,
                  device=self.device,
                  train_loader=self.train_loader,
                  optimizer=self.optimizer,
                  criterion=self.criterion)
        self.optimizer.param_groups[0]["lr"] *= self.lr_decay
        return self.client_id, len(self.train_loader.dataset), self.client_model.state_dict()

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
