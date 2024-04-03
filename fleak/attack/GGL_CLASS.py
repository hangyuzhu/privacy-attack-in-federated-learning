import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import numpy as np
from tqdm import tqdm
import nevergrad as ng



class GGL_reconstruction():

    def __init__(self, fl_model, generator,loss_fn, numclass=10, search_dim=(100,),budget=500, use_tanh=True):
        self.generator = generator
        self.budget = budget
        self.search_dim = search_dim
        self.use_tanh = use_tanh
        self.num_samples = 50
        self.weight = None
        self.loss_fn = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

        parametrization = ng.p.Array(init=np.random.rand(search_dim[0]))
        self.optimizer = ng.optimizers.registry['CMA'](parametrization=parametrization, budget=budget)

        self.setting = {'loss_fn':loss_fn,'fl_model':fl_model}


    def evaluate_loss(self, z, labels, input_gradients, device):
        return self.ng_loss(z=z, input_gradients=input_gradients, device=device, metric='l2', labels=labels, generator=self.generator, weight=self.weight, use_tanh=self.use_tanh, **self.setting)

    def reconstruct(self, input_gradients, device, use_pbar=True):
        labels = self.infer_label(input_gradients)
        print('Inferred label: {}'.format(labels))

        pbar = tqdm(range(self.budget)) if use_pbar else range(self.budget)

        for r in pbar:
            ng_data = [self.optimizer.ask() for _ in range(self.num_samples)]
            # noise = torch.randn(1, 100)
            loss = [self.evaluate_loss(z=ng_data[i].value, labels=labels, input_gradients=input_gradients, device=device) for i in
                    range(self.num_samples)]
            for z, l in zip(ng_data, loss):
                self.optimizer.tell(z, l)

        if use_pbar:
            pbar.set_description("Loss {:.6}".format(np.mean(loss)))
        else:
            print("Round {} - Loss {:.6}".format(r, np.mean(loss)))

        recommendation = self.optimizer.provide_recommendation()
        z_res = torch.Tensor(recommendation.value).unsqueeze(0).to(device)

        if self.use_tanh:
            z_res = z_res.tanh()
        with torch.no_grad():
            x_res = self.generator(z_res)
        # x_res = nn.functional.interpolate(x_res, size=(224, 224), mode='area')


        return x_res, labels

    @staticmethod
    def infer_label(input_gradient):
        label_pred = torch.argmin(torch.sum(list(input_gradient.values())[-2], dim=-1), dim=-1).detach().reshape((1,))
        return label_pred

    @staticmethod
    def ng_loss(z, loss_fn, input_gradients, labels, generator, fl_model, device, metric='l2', use_tanh=True,weight=None):
        z = torch.Tensor(z).unsqueeze(0).to(device)
        if use_tanh:
            z = z.tanh()
        with torch.no_grad():
            x = generator(z)

        target_loss = loss_fn(fl_model(x), labels)
        trial_gradient = torch.autograd.grad(target_loss, fl_model.parameters())
        trial_gradient = [grad.detach() for grad in trial_gradient]

        if weight is not None:
            assert len(weight) == len(trial_gradient)
        else:
            weight = [1] * len(trial_gradient)

        # calculate l2 norm
        dist = 0
        for i in range(len(trial_gradient)):
            if metric == 'l2':
                dist += ((trial_gradient[i] - list(input_gradients.values())[i]).pow(2)).sum() * weight[i]
                # dist += ((trial_gradient[i] - list(input_gradients)[i]).pow(2)).sum() * weight[i]
            elif metric == 'l1':
                dist += ((trial_gradient[i] - list(input_gradients.values())[i]).abs()).sum() * weight[i]


        dist /= len(trial_gradient)

        if not use_tanh:
            KLD = -0.5 * torch.sum(1 + torch.log(torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(z.squeeze(),axis=-1).pow(2) - torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2))
            dist += 0.1 * KLD
        return dist.item()