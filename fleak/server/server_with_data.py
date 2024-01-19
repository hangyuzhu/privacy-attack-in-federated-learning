import copy
import torch
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt


from .server import Server
from fleak.attack import GRNN

class ServerGRNN(Server):
    def __init__(self,
                 server_id=None,
                 server_group=None,
                 global_model=None,
                 momentum=0.0,
                 device=None,
                 num_classes = 0,
                 train_loader=None,
                 valid_loader=None,
                 test_loader=None,
                 img_shape = None):
        super(ServerGRNN, self).__init__(server_id=server_id,
                                        server_group=server_group,
                                        global_model=global_model,
                                        momentum=momentum,
                                        device=device)
        self.num_classes = num_classes
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.img_shape = img_shape


    def comp_grads(self, weights: OrderedDict):
        o_weights = self.global_model.state_dict()
        grads = OrderedDict()
        for (key, value) in weights.items():
            grads[key] = o_weights[key] - weights[key]
            # grads[key].requires_grad = False
        return grads

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

            # convert to gradients
            grads = self.comp_grads(update)
            # update client round
            self.updates.append((c_id, num_samples, update, grads))
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

    def GRNNattack(self):
        reconstruct_data, reconstruct_label = GRNN.Reconstruction(self.num_classes,self.global_model, self.img_shape, self.train_loader)
        return reconstruct_data,reconstruct_label