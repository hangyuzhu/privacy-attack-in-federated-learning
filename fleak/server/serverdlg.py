import torch

from .server import Server
from ..attack.DLG import dlg, idlg
from ..attack.inverting import ig, ig_weight, ig_multiple
from ..attack.robbing_the_fed import robbing
from ..model.gan_network import MnistGenerator, Cifar10Generator
from ..attack.GGL import GGLreconstruction


class ServerDLG(Server):

    def __init__(
        self,
        server_id=None,
        server_group=None,
        global_model=None,
        momentum=0.0,
        dummy_data=None,
        test_loader=None,
        device=None
    ):
        super(ServerDLG, self).__init__(
            server_id=server_id,
            server_group=server_group,
            global_model=global_model,
            momentum=momentum,
            test_loader=test_loader,
            device=device
        )
        self.dummy_data = dummy_data

    def random_attack(self, method="DLG"):
        """
        Randomly select a client to attack from scratch

        :param method: attack method
        :return: reconstructed data and labels
        """
        reconstruct_data = None
        reconstruct_label = None

        local_grads = self.cal_diff(self.updates[0][-1])
        # update global model
        self.global_model.load_state_dict(self.updates[0][-1])

        if method == "DLG":
            reconstruct_data = dlg(
                self.global_model, local_grads, self.dummy_data, self.dummy_labels, 300, 1., self.device)
        elif method == "iDLG":
            reconstruct_data, reconstruct_label = idlg(
                self.global_model, local_grads, self.dummy_data, 300, 0.25)
        elif method == "inverting-gradient":
            reconstruct_data, reconstruct_label = ig_weight(self.global_model, local_grads, self.device)
        elif method == "GGL":
            path = r'D:\leakage-attack-in-federated-learning\models_parameter\cifarGGLgenerator.pth'
            generator = Cifar10Generator().to(self.device)
            generator.load_state_dict(torch.load(path))
            generator.eval()
            reconstruct_data, reconstruct_label = GGLreconstruction(self.global_model, generator, self.updates[0][-1],
                                                                    self.device)
            # path = r'D:\leakage-attack-in-federated-learning\models_parameter\GAN.pth'
            # generator = MnistGenerator().to(self.device)
            # generator.load_state_dict(torch.load(path)['state_dict'])
            # generator.eval()
            # reconstruct_data, reconstruct_label = GGLreconstruction(self.global_model, generator, self.updates[0][-1], self.device)
        elif method == "GRNN":
            reconstruct_data, reconstruct_label = dlg(local_grads, self.dummy_data, self.dummy_labels, self.global_model, 300, 0.001)
        elif method == "Robbing":
            reconstruct_data, reconstruct_label = robbing(self.global_model, local_grads, self.secrets, self.device)
        return reconstruct_data, reconstruct_label

    def fixed_attack(self, method="DLG"):
        # attack the first client
        for update in self.updates:
            if update[0] == 0:
                local_grads = update[-1]
                if method == "DLG":
                    self.dummy_data = dlg(
                        self.global_model, local_grads, self.dummy_data, self.dummy_labels, 300, 1., self.device)
                elif method == "iDLG":
                    reconstruct_data, reconstruct_label = idlg(self.updates[0][-1], self.dummy_data,
                                                               self.dummy_labels, self.global_model, 300,
                                                               0.25)
                elif method == "inverting-gradient":
                    reconstruct_data, reconstruct_label = ig(self.global_model, self.updates[0][-1],
                                                             self.dummy_data)
                elif method == "GGL":
                    path = r'D:\leakage-attack-in-federated-learning\models_parameter\GAN.pth'
                    generator = MnistGenerator().to(self.device)
                    generator.load_state_dict(torch.load(path)['state_dict'])
                    generator.eval()
                    reconstruct_data, reconstruct_label = GGLreconstruction(self.global_model, generator,
                                                                            self.updates[0][-1])
                    # noise = torch.randn(1, 100).to(device)
                    # dummy_data = generator(noise).detach()
                    # reconstruct_data, reconstruct_label = reconstruct_dlg(self.updates[0][-1], dummy_data, self.dummy_labels, self.global_model, 300, 0.001)
                elif method == "GRNN":
                    reconstruct_data, reconstruct_label = dlg(self.updates[0][-1], self.dummy_data,
                                                              self.dummy_labels, self.global_model, 300,
                                                              0.001)
                break

    # def federated_averaging(self):
    #     total_samples = np.sum([update[1] for update in self.updates])
    #     averaged_soln = copy.deepcopy(self.global_model.state_dict())
    #     for key in self.global_model.state_dict().keys():
    #         if 'num_batches_tracked' in key:
    #             continue
    #         for i in range(len(self.updates)):
    #             # global model minus averaged gradients
    #             averaged_soln[key] -= self.updates[i][2][key] * self.updates[i][1] / total_samples
    #     self.accumulate_momentum(averaged_soln)
    #     # update global model
    #     self.global_model.load_state_dict(averaged_soln)
    #     # clear uploads buffer
    #     self.updates = []
