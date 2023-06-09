import copy

from .server import Server
from collections import OrderedDict


class ServerDLG(Server):

    def __init__(self,
                 server_id=None,
                 server_group=None,
                 global_model=None,
                 momentum=0.0,
                 data_size=None,
                 device=None):
        super(ServerDLG, self).__init__(server_id=server_id,
                                        server_group=server_group,
                                        global_model=global_model,
                                        momentum=momentum,
                                        device=device)
        self.data_size = data_size
        self.dummy_data = 
        self.dummy_labels =

    def comp_grads(self, weights: OrderedDict):
        o_weights = self.global_model.state_dict()
        grads = OrderedDict()
        for (key, value) in weights.items():
            grads[key] = o_weights[key] - weights[key]
            grads[key].requires_grad = False
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
            self.updates.append((c_id, num_samples, grads))
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

    def attack(self, method = "DLG"):

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
        self.accumulate_momentum(averaged_soln)
        # update global model
        self.global_model.load_state_dict(averaged_soln)
        # clear uploads buffer
        self.updates = []