import copy

import torch
import torch.optim as optim
import torch.nn as nn
from ..utils.train_eval import train, evaluate

from .client import Client
from fleak.attack import GAN

device = "cuda" if torch.cuda.is_available() else "CPU"

class Maliciousclient(Client):
    def __init__(self,
                 client_id=None,
                 client_group=None,
                 client_model=None,
                 num_epochs=1,
                 img_size = None,
                 lr=0.1,
                 lr_decay=0.95,
                 momentum=0.5,
                 train_loader=None,
                 valid_loader=None,
                 test_loader=None,
                 device=None):
        super(Maliciousclient,self).__init__(client_id=client_id,
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
        self.img_size = img_size

    def synchronize(self, cur_round, model_params):
        self.cur_round = cur_round
        # inner deep copied
        self.client_model.load_state_dict(model_params)

    def train(self):
        for local_epoch in range(self.num_epochs):
            # local batch training
            train(model=self.client_model,
                  device=self.device,
                  train_loader=self.train_loader,
                  optimizer=self.optimizer,
                  criterion=self.criterion)
        self.optimizer.param_groups[0]["lr"] *= self.lr_decay
        ## 训练完开始进行GAN攻击，令id=0的客户端为恶意客户端
        if self.client_id == 0:
            label_a = torch.randint(0, 9, (len(self.train_loader.dataset), 1))
            gen_image, gen_label=GAN.GAN_attack(self.client_model, batch=64, img_size=self.img_size, attack_label=label_a, dataset=self.train_loader.dataset)
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

    def save(self, path):
        torch.save(self.client_model.state_dict(), path)



