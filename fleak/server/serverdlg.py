import copy
import torch
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

from .server import Server
from fleak.attack.idlg import dlg, idlg
from fleak.attack.inverting import ig, ig_weight, ig_multiple
from fleak.attack.robbing_the_fed import robbing
from fleak.model.gan_network import MnistGenerator,Cifar10Generator
from fleak.attack.GGL import GGLreconstruction



class ServerDLG(Server):

    def __init__(self,
                 server_id=None,
                 server_group=None,
                 global_model=None,
                 momentum=0.0,
                 data_size=None,
                 label_size=None,
                 secrets = None,
                 device=None,):
        super(ServerDLG, self).__init__(server_id=server_id,
                                        server_group=server_group,
                                        global_model=global_model,
                                        momentum=momentum,
                                        device=device)
        self.data_size = data_size
        self.label_size = label_size
        self.secrets = secrets
        self.dummy_data = torch.randn(data_size).to(device).requires_grad_(True)
        self.dummy_labels = torch.randn(label_size).to(device).requires_grad_(True)

    def comp_grads(self, weights: OrderedDict):
        o_weights = self.global_model.state_dict()
        grads = OrderedDict()
        for (key, value) in weights.items():
            grads[key] = o_weights[key] - weights[key]
            # grads[key].requires_grad = False
        return grads

    # def train_eval(self, clients=None, set_to_use='test'):
    #     if clients is None:
    #         clients = self.selected_clients
    #     eval_correct = 0
    #     eval_total = 0
    #     for c in clients:
    #         c.synchronize(self.cur_round, self.global_model.state_dict())
    #         eval_cor, eval_tot = c.evaluate(set_to_use=set_to_use)
    #         eval_correct += eval_cor
    #         eval_total += eval_tot
    #         # train and update client model
    #         c_id, num_samples, update = c.train()
    #
    #         # convert to gradients
    #         # grads = self.comp_grads(update)
    #         # update client round
    #         # self.updates.append((c_id, num_samples, grads))
    #         self.updates.append((c_id, num_samples, update))
    #     eval_accuracy = eval_correct / eval_total
    #     print('Round %d: ' % self.cur_round + set_to_use + ' accuracy %.4f' % eval_accuracy)
    #     # update communication round
    #     self.cur_round += 1
    #     return eval_accuracy

    def train(self, clients=None):
        # just for training
        if clients is None:
            clients = self.selected_clients
        for c in clients:
            c.synchronize(self.cur_round, self.global_model.state_dict())
            # train and update client model
            c_id, num_samples, update = c.train()
            # gather client uploads into a buffer
            self.updates.append((c_id, num_samples, update))
        # update communication round
        self.cur_round += 1

    def random_attack(self, method="DLG"):
        """
        Randomly select a client to attack from scratch
        :param method: attack method
        :return: reconstructed data and labels
        """
        reconstruct_data = None
        reconstruct_label = None

        local_grads = self.comp_grads(self.updates[0][-1])
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
            path = r'D:\leakage-attack-in-federated-learning\models_parameter\GAN.pth'
            generator = MnistGenerator().to(self.device)
            generator.load_state_dict(torch.load(path)['state_dict'])
            generator.eval()
            reconstruct_data, reconstruct_label = GGLreconstruction(self.global_model, generator, self.updates[0][-1], self.device)
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
