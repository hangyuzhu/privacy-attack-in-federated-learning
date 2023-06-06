import numpy as np
import torch
import copy


class Server:

    def __init__(self,
                 server_id=None,
                 server_group=None,
                 global_model=None,
                 momentum=0.0,
                 device=None):
        self.server_id = server_id
        self.server_group = server_group
        self.device = device
        self.global_model = global_model.to(self.device)
        self.momentum = momentum
        self.momentum_buffer = None
        self._init_momentum()
        self.selected_clients = None
        self.updates = []
        self.cur_round = 0

    @property
    def model_size(self):
        return sum([p.numel() * p.element_size()
                    for p in self.global_model.state_dict().values()]) / 1e6

    def _init_momentum(self):
        if self.momentum:
            self.momentum_buffer = copy.deepcopy(self.global_model.state_dict())
            for key in self.momentum_buffer:
                self.momentum_buffer[key] = 0

    def select_clients(self, possible_clients, num_clients=20):
        num_clients = min(num_clients, len(possible_clients))
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

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
            num_samples, update = c.train()
            # update client round
            self.updates.append((num_samples, update))
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
            num_samples, update = c.train()
            # gather client uploads into a buffer
            self.updates.append((num_samples, update))
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

    def accumulate_momentum(self, averaged_soln):
        # momentum is applied on the server
        if self.momentum_buffer is not None:
            for key in self.momentum_buffer.keys():
                if 'running_' in key:
                    continue
                delta_w = self.global_model.state_dict()[key] - averaged_soln[key]
                self.momentum_buffer[key] = self.momentum * self.momentum_buffer[key] + (1 - self.momentum) * delta_w
                averaged_soln[key] = self.global_model.state_dict()[key] - self.momentum_buffer[key]

    def federated_averaging(self):
        total_samples = np.sum([update[0] for update in self.updates])
        averaged_soln = copy.deepcopy(self.updates[0][1])
        for key in self.global_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                continue
            for i in range(len(self.updates)):
                if i == 0:
                    averaged_soln[key] = averaged_soln[key] * self.updates[i][0] / total_samples
                else:
                    averaged_soln[key] += self.updates[i][1][key] * self.updates[i][0] / total_samples
        self.accumulate_momentum(averaged_soln)
        # update global model
        self.global_model.load_state_dict(averaged_soln)
        # clear uploads buffer
        self.updates = []

    def save_model(self, path):
        # Save server model
        torch.save(self.global_model.state_dict(), path)
