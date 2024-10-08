import torch
import torch.optim as optim
import torch.nn as nn
from .wrapper import dmgan_class
from ..utils.train_eval import train, evaluate


class Client:

    def __init__(
        self,
        client_id=None,
        client_group=None,
        client_model=None,
        local_epochs=1,
        lr=0.1,
        lr_decay=0.95,
        momentum=0.5,
        train_loader=None,
        valid_loader=None,
        test_loader=None,
        device=None
    ):
        self.client_id = client_id
        self.client_group = client_group
        self.device = device
        self.cur_round = 0
        self.client_model = client_model.to(self.device)
        self.num_epochs = local_epochs
        self.optimizer = optim.SGD(self.client_model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-5)
        self.lr_decay = lr_decay    # lr decay for each FL round
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

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


@dmgan_class
class GanClient(Client):
    """

    Adversarial client adopting GAN attack

    """

    def __init__(self,
                 client_id=None,
                 client_group=None,
                 client_model=None,
                 generator=None,
                 tracked_class=3,
                 local_epochs=1,
                 rec_epochs=1,
                 lr=0.1,
                 lr_decay=0.95,
                 momentum=0.5,
                 train_loader=None,
                 valid_loader=None,
                 test_loader=None,
                 dummy=None,
                 device=None):
        super(GanClient, self).__init__(
            client_id=client_id,
            client_group=client_group,
            client_model=client_model,
            local_epochs=local_epochs,
            lr=lr,
            lr_decay=lr_decay,
            momentum=momentum,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            device=device
        )
        self.tracked_class = tracked_class

        self.generator = generator.to(self.device)
        self.d_optimizer = optim.SGD(self.client_model.parameters(), lr=1e-3, weight_decay=1e-7)
        self.g_optimizer = optim.SGD(self.generator.parameters(), lr=1e-4, weight_decay=1e-7)

        self.rec_epochs = rec_epochs
        self.dummy = dummy
