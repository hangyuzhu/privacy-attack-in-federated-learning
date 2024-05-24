import copy
import numpy as np
import torch

from ..attack import dlg, idlg
from ..attack import ig_single, ig_multi
from ..attack import invert_linear_layer
from ..attack import grnn
from ..attack import ggl
from ..attack import cpa


class Server:

    def __init__(
        self,
        server_id=None,
        server_group=None,
        global_model=None,
        test_loader=None,
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

    @property
    def model_size(self):
        return sum([p.numel() * p.element_size() for p in self.global_model.state_dict().values()])

    def select_clients(self, possible_clients, num_clients=20):
        num_clients = min(num_clients, len(possible_clients))
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

    def extract_gradients(self, local_params):
        """ Extract the gradients of any client model

        Using named_parameters to avoid incorrect computation with running statistics
        Caution: .detach() is adopted here to cut off the grad_fn

        :param local_params: client model parameters
        :return:
        """
        diffs = [(v - local_params[k]).detach() for k, v in self.global_model.named_parameters()]
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

    def save_model(self, path):
        # Save server model
        torch.save(self.global_model.state_dict(), path)


class ServerAttacker(Server):

    def __init__(
        self,
        server_id=None,
        server_group=None,
        global_model=None,
        generator=None,
        test_loader=None,
        dummy=None,
        local_epochs=1,
        local_lr=0.1,
        device=None
    ):
        super().__init__(
            server_id=server_id,
            server_group=server_group,
            global_model=global_model,
            test_loader=test_loader,
            device=device
        )
        if generator is not None:
            self.generator = generator.to(self.device)

        self.dummy = dummy
        self.local_epochs = local_epochs
        self.local_lr = local_lr

    def attack(self, method):
        """
        Randomly select a client to infer its private data

        :param method: attack method
        :return: reconstructed data and labels
        """
        local_grads = self.extract_gradients(self.updates[0][-1])
        # replace the global model by client model
        self.global_model.load_state_dict(self.updates[0][-1])

        if method == "dlg":
            dlg(self.global_model, local_grads, self.dummy, 300, self.device)
        elif method == "idlg":
            idlg(self.global_model, local_grads, self.dummy, 300, 1.0, self.device)
        elif method == "ig_single":
            ig_single(self.global_model, local_grads, self.dummy, 4000, 0.1, 1e-6, self.device)
        elif method == "ig_multi":
            ig_multi(
                self.global_model, local_grads, self.dummy,
                8000, 0.1, self.local_epochs, self.local_lr, 1e-6, self.device)
        elif method == "rtf":
            invert_linear_layer(local_grads, self.dummy)
        elif method == "ggl":
            ggl(self.global_model, self.generator, local_grads, self.dummy, 25000, self.device)
        elif method == "grnn":
            grnn(self.global_model, local_grads, self.dummy, 1000, 1e-3, self.device)
        elif method == "cpa":
            cpa(self.global_model, local_grads, self.dummy, 25000, 0.001, 5.3, 7.7, 0.1, 0.13, 5, 1, self.device)
        else:
            raise ValueError("Unexpected {} Attack Type.".format(method))