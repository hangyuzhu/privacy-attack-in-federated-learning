import torch

from .server import Server
from ..attack.cocktail import cocktail
from collections import OrderedDict
from ..attack.Robbing import robbing
from ..model.gan_network import MnistGenerator, Cifar10Generator
from ..attack.GGL import GGLreconstruction


class ServerCock(Server):

    def __init__(
        self,
        server_id=None,
        server_group=None,
        global_model=None,
        test_loader=None,
        device=None,
        datasets = "tiny_imagenet",
        batch_size = 1,
        fc_index = 0
    ):
        super(ServerCock, self).__init__(
            server_id=server_id,
            server_group=server_group,
            global_model=global_model,
            test_loader=test_loader,
            device=device
        )
        self.ds = datasets
        self.batch_size = batch_size
        self.fc_index = fc_index


    def random_attack(self):
        """
        Randomly select a client to attack from scratch

        :param method: attack method
        :return: reconstructed data and labels
        """
        # reconstruct_data = None
        # reconstruct_label = None

        local_grads = self._subtract(self.global_model.state_dict(), self.updates[0][-1])
        # update global model
        self.global_model.load_state_dict(self.updates[0][-1])

        reconstruct_data = cocktail(self.global_model,local_grads,self.batch_size,self.fc_index,lr=0.001,device=self.device)

        return reconstruct_data

    # def fixed_attack(self, method="DLG"):
    #     # attack the first client
    #     for update in self.updates:
    #         if update[0] == 0:
    #             local_grads = update[-1]
    #             if method == "DLG":
    #                 self.dummy_data = dlg(
    #                     self.global_model, local_grads, self.dummy_data, self.dummy_labels, 300, 1., self.device)
    #             elif method == "iDLG":
    #                 reconstruct_data, reconstruct_label = idlg(self.updates[0][-1], self.dummy_data,
    #                                                            self.dummy_labels, self.global_model, 300,
    #                                                            0.25)
    #             elif method == "inverting-gradient":
    #                 reconstruct_data, reconstruct_label = ig(self.global_model, self.updates[0][-1],
    #                                                          self.dummy_data)
    #             elif method == "GGL":
    #                 path = r'D:\leakage-attack-in-federated-learning\models_parameter\GAN.pth'
    #                 generator = MnistGenerator().to(self.device)
    #                 generator.load_state_dict(torch.load(path)['state_dict'])
    #                 generator.eval()
    #                 reconstruct_data, reconstruct_label = GGLreconstruction(self.global_model, generator,
    #                                                                         self.updates[0][-1])
    #                 # noise = torch.randn(1, 100).to(device)
    #                 # dummy_data = generator(noise).detach()
    #                 # reconstruct_data, reconstruct_label = reconstruct_dlg(self.updates[0][-1], dummy_data, self.dummy_labels, self.global_model, 300, 0.001)
    #             elif method == "GRNN":
    #                 reconstruct_data, reconstruct_label = dlg(self.updates[0][-1], self.dummy_data,
    #                                                           self.dummy_labels, self.global_model, 300,
    #                                                           0.001)
    #             break