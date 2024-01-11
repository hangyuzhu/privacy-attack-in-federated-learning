import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import numpy as np
from tqdm import tqdm
import nevergrad as ng
from fleak.utils import loss_fn

device = "cuda" if torch.cuda.is_available() else "CPU"


def infer_label(input_gradient):
    label_pred = torch.argmin(torch.sum(list(input_gradient.values())[-4])).detach().reshape((1,))
    return label_pred


def ng_loss(z, input_gradient, labels, generator, fl_model,num_classes = 10,metric='l2',use_tanh=True,weight=None):
    # z = torch.Tensor(z).unsqueeze(0).to(device)

    if use_tanh:
        z = z.tanh()

    # c = torch.nn.functional.one_hot(labels, num_classes=num_classes).to(device)

    with torch.no_grad():
        x = generator(z)
    loss_fn = nn.CrossEntropyLoss()
    # x = nn.functional.interpolate(x, size=(224, 224), mode='area')

    # compute the trial gradient
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
            dist += ((trial_gradient[i] - list(input_gradient.values())[i]).pow(2)).sum() * weight[i]
        elif metric == 'l1':
            dist += ((trial_gradient[i] - list(input_gradient.values())[i]).abs()).sum() * weight[i]


    dist /= len(trial_gradient)

    if not use_tanh:
        KLD = -0.5 * torch.sum(1 + torch.log(torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(z.squeeze(),axis=-1).pow(2) - torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2))
        dist += 0.1 * KLD
    return dist.item()


def GGLreconstruction(global_model, generator, shared_gradients,budget = 500, use_tanh=True):
    dummy_label = infer_label(shared_gradients)
    noise = torch.randn(1, 100).to(device)
    # dummy_data = generator(noise).detach()
    pbar = tqdm(range(budget))
    parametrization = ng.p.Array(init=np.random.rand(128))
    optimizer = ng.optimizers.registry['CMA'](parametrization=parametrization, budget=budget)

    for _ in pbar:
        # ng_data = [optimizer.ask() for _ in range(50)]
        # loss = [ng_loss(z=ng_data[i].value, labels=dummy_label, input_gradient=shared_gradients, metric='l2',generator=generator, weight=None, use_tanh=use_tanh,loss_fn=loss_fn,fl_model=global_model,num_classes=10) for i in
        #         range(50)]
        noise = torch.randn(1, 100).to(device)
        for _ in range(50):
            loss = [ng_loss(z=noise, labels=dummy_label, input_gradient=shared_gradients, metric='l2',generator=generator, weight=None, use_tanh=use_tanh,fl_model=global_model,num_classes=10)]
            # for z, l in zip(ng_data, loss):
            #     optimizer.tell(z, l)
            optimizer.tell(noise, loss)

        pbar.set_description("Loss {:.6}".format(np.mean(loss)))

    recommendation = optimizer.provide_recommendation()
    z_res = torch.from_numpy(recommendation.value).unsqueeze(0).to(device)
    if use_tanh:
        z_res = z_res.tanh()
    loss_res = ng_loss(z=recommendation.value, input_gradient=shared_gradients, metric='l2', labels=dummy_label,generator=generator, weight=None, use_tanh=use_tanh,fl_model=global_model,num_classes=10)
    with torch.no_grad():
        # x_res = generator(z_res.float(), dummy_label.float(), 1)
        x_res = generator(z_res.float())
    x_res = nn.functional.interpolate(x_res, size=(224, 224), mode='area')
    print(loss_res)

    return x_res, dummy_label