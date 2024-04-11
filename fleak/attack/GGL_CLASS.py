import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import numpy as np
from tqdm import tqdm
import nevergrad as ng
from fleak.utils.turbo import Turbo1


class GGL_reconstruction:

    def __init__(self, fl_model, generator,loss_fn, numclass=10, search_dim=(100,), budget=500, use_tanh=True):
        self.generator = generator
        self.budget = budget
        self.search_dim = search_dim
        self.use_tanh = use_tanh
        self.num_samples = 50
        self.weight = None
        self.loss_fn = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

        parametrization = ng.p.Array(init=np.random.rand(search_dim[0]))
        self.optimizer = ng.optimizers.registry['CMA'](parametrization=parametrization, budget=budget)

        self.setting = {'loss_fn': loss_fn, 'fl_model': fl_model}

    def evaluate_loss(self, z, labels, input_gradients, device):
        return self.ng_loss(z=z, input_gradients=input_gradients, device=device, metric='l2', labels=labels, generator=self.generator, weight=self.weight, use_tanh=self.use_tanh, **self.setting)

    def reconstruct(self, input_gradients, device, use_pbar=True):
        # NOTE
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

    # @staticmethod
    # def infer_label(input_gradient):
    #     label_pred = torch.argmin(torch.sum(list(input_gradient.values())[-2], dim=-1), dim=-1).detach().reshape((1,))
    #     return label_pred

    @staticmethod
    def infer_label(input_gradient, num_inputs=1):
        last_weight_min = torch.argsort(torch.sum(input_gradient[-2], dim=-1), dim=-1)[:num_inputs]
        labels = last_weight_min.detach().reshape((-1,)).requires_grad_(False)
        return labels

    @staticmethod
    def ng_loss(z,
                loss_fn,
                input_gradients,
                labels,
                generator,
                fl_model,
                device,
                metric='l2',
                use_tanh=True,
                weight=None):

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


class BOReconstructor:
    """
    Bayesian optimization
    """
    def __init__(self, fl_model, generator, loss_fn, search_dim=(100,), budget=500, use_tanh=True, use_weight=False):

        self.generator = generator
        self.budget = budget
        self.search_dim = search_dim
        self.use_tanh = use_tanh
        self.num_samples = 50
        self.weight = None

        self.fl_setting = {'loss_fn':loss_fn, 'fl_model':fl_model}

        if use_weight:
            self.weight = np.ones(62,)
            for i in range(0, 20):
                self.weight[3*i:3*(i+1)] /= 2**i


    def evaluate_loss(self, z, labels, input_gradient,device):
        return self.ng_loss(z=z, input_gradient=input_gradient, metric='l2',
                        labels=labels, generator=self.generator, weight=self.weight, device=device,
                        use_tanh=self.use_tanh, **self.fl_setting
                       )

    def reconstruct(self, input_gradient, device, use_pbar=True):

        labels = self.infer_label(input_gradient)
        print('Inferred label: {}'.format(labels))

        z_lb = -2*np.ones(self.search_dim) # lower bound, you may change -10 to -inf
        z_ub = 2*np.ones(self.search_dim) # upper bound, you may change 10 to inf

        f = lambda z:self.evaluate_loss(z, labels, input_gradient, device)

        self.optimizer = Turbo1(
                                f=f,  # Handle to objective function
                                lb=z_lb,  # Numpy array specifying lower bounds
                                ub=z_ub,  # Numpy array specifying upper bounds
                                n_init=256,  # Number of initial bounds from an Latin hypercube design
                                max_evals=self.budget,  # Maximum number of evaluations
                                batch_size=10,  # How large batch size TuRBO uses
                                verbose=True,  # Print information from each batch
                                use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                                max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                                n_training_steps=50,  # Number of steps of ADAM to learn the hypers
                                min_cuda=1024,  # Run on the CPU for small datasets
                                device="cuda", #next(generator.parameters()).device,  # "cpu" or "cuda"
                                dtype="float32",  # float64 or float32
                            )

        self.optimizer.optimize()

        X = self.optimizer.X  # Evaluated points of z
        fX = self.optimizer.fX  # Observed values of ng_loss
        ind_best = np.argmin(fX)
        loss_res, z_res = fX[ind_best], X[ind_best, :]

        # loss_res = self.evaluate_loss(z_res, labels, input_gradient)
        z_res = torch.from_numpy(z_res).unsqueeze(0).to(device)
        if self.use_tanh:
            z_res = z_res.tanh()
        z_res = z_res.to(torch.float32)
        with torch.no_grad():
            x_res = self.generator(z_res)

        return x_res, labels

    @staticmethod
    def infer_label(input_gradient, num_inputs=1):
        label_pred = torch.argmin(torch.sum(list(input_gradient.values())[-2], dim=-1), dim=-1).detach().reshape((1,))
        # last_weight_min = torch.argsort(torch.sum(input_gradient[-2], dim=-1), dim=-1)[:num_inputs]
        # labels = last_weight_min.detach().reshape((-1,)).requires_grad_(False)
        return label_pred

    @staticmethod
    def ng_loss(z, # latent variable to be optimized
                loss_fn, # loss function for FL model
                input_gradient,
                labels,
                generator,
                fl_model,
                device,
                metric='l2',
                use_tanh=True,
                weight=None, # weight to be applied when calculating the gradient matching loss
               ):

        z = torch.Tensor(z).unsqueeze(0).to(device)
        if use_tanh:
            z = z.tanh()
        with torch.no_grad():
            x = generator(z)

        # compute the trial gradient
        target_loss = loss_fn(fl_model(x), labels)
        trial_gradient = torch.autograd.grad(target_loss, fl_model.parameters())
        trial_gradient = [grad.detach() for grad in trial_gradient]


        if weight is not None:
            assert len(weight) == len(trial_gradient)
        else:
            weight = [1]*len(trial_gradient)

        # calculate l2 norm
        dist = 0
        for i in range(len(trial_gradient)):
            if metric == 'l2':
                dist += ((trial_gradient[i] - list(input_gradient.values())[i]).pow(2)).sum() * weight[i]
                # dist += ((trial_gradient[i] - list(input_gradients)[i]).pow(2)).sum() * weight[i]
            elif metric == 'l1':
                dist += ((trial_gradient[i] - list(input_gradient.values())[i]).abs()).sum() * weight[i]
        dist /= len(trial_gradient)

        if not use_tanh:
            KLD = -0.5 * torch.sum(1 + torch.log(torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(z.squeeze(), axis=-1).pow(2) - torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2))
            dist += 0.1*KLD

        return dist.item()
# class BOReconstructor():
#     """
#     Bayesian optimization
#     """
#     def __init__(self, fl_model, generator, loss_fn, num_classes=10, search_dim=(100,), budget=500, use_tanh=False, use_weight=False):
#
#         self.generator = generator
#         self.budget = budget
#         self.search_dim = search_dim
#         self.use_tanh = use_tanh
#         self.num_samples = 50
#         self.weight = None
#
#         self.fl_setting = {'loss_fn':loss_fn, 'fl_model':fl_model, 'num_classes':num_classes}
#
#         if use_weight:
#             self.weight = np.ones(62,)
#             for i in range(0, 20):
#                 self.weight[3*i:3*(i+1)] /= 2**i
#
#
#     def evaluate_loss(self, z, labels, input_gradient):
#         return self.ng_loss(z=z, input_gradient=input_gradient, metric='l2',
#                         labels=labels, generator=self.generator, weight=self.weight,
#                         use_tanh=self.use_tanh, **self.fl_setting
#                        )
#
#     def reconstruct(self, input_gradient, use_pbar=True):
#
#         labels = self.infer_label(input_gradient)
#         print('Inferred label: {}'.format(labels))
#
#         if self.defense_setting is not None:
#             if 'clipping' in self.defense_setting:
#                 total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in input_gradient]), 2)
#                 self.defense_setting['clipping'] = total_norm.item()
#                 print('Estimated defense parameter: {}'.format(self.defense_setting['clipping']))
#             if 'compression' in self.defense_setting:
#                 n_zero, n_param = 0, 0
#                 for i in range(len(input_gradient)):
#                     n_zero += torch.sum(input_gradient[i]==0)
#                     n_param += torch.numel(input_gradient[i])
#                 self.defense_setting['compression'] = 100 * (n_zero/n_param).item()
#                 print('Estimated defense parameter: {}'.format(self.defense_setting['compression']))
#
#         c = torch.nn.functional.one_hot(labels, num_classes=self.fl_setting['num_classes']).to(input_gradient[0].device)
#
#
#
#         z_lb = -2*np.ones(self.search_dim) # lower bound, you may change -10 to -inf
#         z_ub = 2*np.ones(self.search_dim) # upper bound, you may change 10 to inf
#
#         f = lambda z:self.evaluate_loss(z, labels, input_gradient)
#
#         self.optimizer = Turbo1(
#                                 f=f,  # Handle to objective function
#                                 lb=z_lb,  # Numpy array specifying lower bounds
#                                 ub=z_ub,  # Numpy array specifying upper bounds
#                                 n_init=256,  # Number of initial bounds from an Latin hypercube design
#                                 max_evals=self.budget,  # Maximum number of evaluations
#                                 batch_size=10,  # How large batch size TuRBO uses
#                                 verbose=True,  # Print information from each batch
#                                 use_ard=True,  # Set to true if you want to use ARD for the GP kernel
#                                 max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
#                                 n_training_steps=50,  # Number of steps of ADAM to learn the hypers
#                                 min_cuda=1024,  # Run on the CPU for small datasets
#                                 device="cuda", #next(generator.parameters()).device,  # "cpu" or "cuda"
#                                 dtype="float32",  # float64 or float32
#                             )
#
#         self.optimizer.optimize()
#
#         X = self.optimizer.X  # Evaluated points of z
#         fX = self.optimizer.fX  # Observed values of ng_loss
#         ind_best = np.argmin(fX)
#         loss_res, z_res = fX[ind_best], X[ind_best, :]
#
#         loss_res = self.evaluate_loss(z_res, labels, input_gradient)
#         z_res = torch.from_numpy(z_res).unsqueeze(0).to(input_gradient[0].device)
#         if self.use_tanh:
#             z_res = z_res.tanh()
#
#         with torch.no_grad():
#             x_res = self.generator(z_res.float(), c.float(), 1)
#         x_res = nn.functional.interpolate(x_res, size=(224, 224), mode='area')
#
#         return x_res, labels
#
#     @staticmethod
#     def infer_label(input_gradient, num_inputs=1):
#         last_weight_min = torch.argsort(torch.sum(input_gradient[-2], dim=-1), dim=-1)[:num_inputs]
#         labels = last_weight_min.detach().reshape((-1,)).requires_grad_(False)
#         return labels
#
#     @staticmethod
#     def ng_loss(z, # latent variable to be optimized
#                 loss_fn, # loss function for FL model
#                 input_gradient,
#                 labels,
#                 generator,
#                 fl_model,
#                 num_classes=1000,
#                 metric='l2',
#                 use_tanh=True,
#                 weight=None, # weight to be applied when calculating the gradient matching loss
#                ):
#
#         z = torch.Tensor(z).unsqueeze(0).to(input_gradient[0].device)
#         if use_tanh:
#             z = z.tanh()
#
#         c = torch.nn.functional.one_hot(labels, num_classes=num_classes).to(input_gradient[0].device)
#
#         with torch.no_grad():
#             x = generator(z, c.float(), 1)
#
#         x = nn.functional.interpolate(x, size=(224, 224), mode='area')
#
#         # compute the trial gradient
#         target_loss, _, _ = loss_fn(fl_model(x), labels)
#         trial_gradient = torch.autograd.grad(target_loss, fl_model.parameters())
#         trial_gradient = [grad.detach() for grad in trial_gradient]
#
#
#         if weight is not None:
#             assert len(weight) == len(trial_gradient)
#         else:
#             weight = [1]*len(trial_gradient)
#
#         # calculate l2 norm
#         dist = 0
#         for i in range(len(trial_gradient)):
#             if metric == 'l2':
#                 dist += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum()*weight[i]
#             elif metric == 'l1':
#                 dist += ((trial_gradient[i] - input_gradient[i]).abs()).sum()*weight[i]
#         dist /= len(trial_gradient)
#
#         if not use_tanh:
#             KLD = -0.5 * torch.sum(1 + torch.log(torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(z.squeeze(), axis=-1).pow(2) - torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2))
#             dist += 0.1*KLD
#
#         return dist.item()





