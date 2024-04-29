import torch
import copy
import numpy as np
from collections import OrderedDict

from ..attack.DLG import dlg, idlg
from ..attack.inverting import ig, ig_weight, ig_multiple
from ..attack.robbing_the_fed import robbing
from ..model.gan_network import MnistGenerator, Cifar10Generator
from ..attack.GGL import GGLreconstruction


class Server:

    def __init__(
        self,
        server_id=None,
        server_group=None,
        global_model=None,
        test_loader=None,
        dummy=None,
        device=None
    ):
        # server info
        self.server_id = server_id
        self.server_group = server_group
        self.device = device
        self.selected_clients = None
        self.updates = []
        self.cur_round = 0
        # model & data
        self.global_model = global_model.to(self.device)
        self.test_loader = test_loader
        self.dummy = dummy

    @property
    def model_size(self):
        return sum([p.numel() * p.element_size() for p in self.global_model.state_dict().values()])

    def select_clients(self, possible_clients, num_clients=20):
        num_clients = min(num_clients, len(possible_clients))
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

    @staticmethod
    def _subtract(global_params, local_params):
        """
        Calculate the difference between the global model and uploaded local model

        :param global_params: parameters of the global model
        :param local_params: parameters of uploaded model
        :return: a dict containing tensor differences
        """
        diffs = OrderedDict()
        for (key, value) in local_params.items():
            diffs[key] = global_params[key] - local_params[key]
        return diffs

    def train_eval(self, clients=None, set_to_use='test'):
        if clients is None:
            clients = self.selected_clients
        eval_correct = 0
        eval_total = 0
        for c in clients:
            c.synchronize(self.cur_round, self.global_model.state_dict())
            eval_cor, eval_tot = c.evaluate(set_to_use=set_to_use)
            eval_correct += eval_cor
            eval_total += eval_tot
            # train and update client model
            c_id, num_samples, update = c.train()
            # update client round
            self.updates.append((c_id, num_samples, update))
        eval_accuracy = eval_correct / eval_total
        print('Round %d: ' % self.cur_round + set_to_use + ' accuracy %.4f' % eval_accuracy)
        # update communication round
        self.cur_round += 1
        return eval_accuracy

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

    def evaluate(self, clients=None, set_to_use='test'):
        if clients is None:
            clients = self.selected_clients
        correct = 0
        total = 0
        for i, c in enumerate(clients):
            c.synchronize(self.cur_round, self.global_model.state_dict())
            corr, tot = c.evaluate(set_to_use=set_to_use)
            correct += corr
            total += tot
        accuracy = correct / total
        print('Round %d: ' % self.cur_round + set_to_use + ' accuracy %.4f' % accuracy)
        return accuracy

    def federated_averaging(self):
        total_samples = np.sum([update[1] for update in self.updates])
        averaged_soln = copy.deepcopy(self.updates[0][2])
        for key in self.global_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                continue
            for i in range(len(self.updates)):
                if i == 0:
                    averaged_soln[key] = averaged_soln[key] * self.updates[i][1] / total_samples
                else:
                    averaged_soln[key] += self.updates[i][2][key] * self.updates[i][1] / total_samples
        # update global model
        self.global_model.load_state_dict(averaged_soln)
        # clear uploads buffer
        self.updates = []

    def attack(self, method="DLG"):
        """
        Randomly select a client to attack from scratch

        :param method: attack method
        :return: reconstructed data and labels
        """
        local_grads = self._subtract(self.global_model.state_dict(), self.updates[0][-1])
        # update global model
        self.global_model.load_state_dict(self.updates[0][-1])

        if method == "DLG":
            dlg(self.global_model, local_grads, self.dummy, 300, self.device)
        elif method == "iDLG":
            idlg(self.global_model, local_grads, self.dummy, 300, 1.0, self.device)
        else:
            raise ValueError("Unexpected {} Attack Type.".format(method))

    def save_model(self, path):
        # Save server model
        torch.save(self.global_model.state_dict(), path)
