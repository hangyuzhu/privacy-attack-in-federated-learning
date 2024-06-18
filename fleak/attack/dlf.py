"""Deep Leakage in Federated Averaging

https://openreview.net/pdf?id=e7A0B99zJf

"""
import copy
import time
import math
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from .ig import total_variation
from ..model import MetaModel


def dlf(model, gt_grads, dummy, labels, rec_epochs, rec_lr, epochs, lr, k_batches, batch_size, tv, reg_clip, reg_reorder, device):
    # dummy data with size = k_batches * batch_size
    dummy_data = dummy.generate_dummy_input(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.Adam([dummy_data], lr=rec_lr)
    start = 4
    alpha = math.exp(1.0 / rec_epochs * math.log(2 / start))
    minmax = 2
    reorder_prior = "l2_max_conv"

    layer_weights = torch.arange(len(gt_grads), 0, -1)
    layer_weights = torch.exp(layer_weights)
    layer_weights = layer_weights / torch.sum(layer_weights)
    layer_weights = layer_weights / layer_weights[0]

    # assume prior is none
    # exp is not adopted
    pbar = tqdm(range(rec_epochs * 2),
                total=rec_epochs * 2,
                desc=f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}')
    curr_fac = start
    for _ in pbar:
        curr_fac = min(curr_fac * alpha, minmax) if alpha >= 1.0 else max(curr_fac * alpha, minmax)

        optimizer.zero_grad()
        model.zero_grad()  # not necessary

        inv_prior = 0
        if reorder_prior is not None:
            inv_prior = order_invariant_prior(dummy_data, reorder_prior, epochs, device)

        dummy_grads = get_dummy_grads(
            model, dummy_data, labels, epochs, lr, k_batches, batch_size, criterion
        )
        grad_diff = 0
        for dummy_g, gt_g, lw in zip(dummy_grads, gt_grads, layer_weights):
            grad_diff += lw * ((dummy_g - gt_g) ** 2).sum()
        grad_diff /= k_batches

        tot_loss = curr_fac * grad_diff + tv * total_variation(dummy_data) + reg_clip * clip_prior(dummy_data, -dummy.t_dm / dummy.t_ds, 1 / dummy.t_ds) + reg_reorder * inv_prior
        tot_loss.backward()
        optimizer.step()
        pbar.set_description("Loss {:.6}".format(tot_loss))

    dummy.append(dummy_data)

    return dummy_data


def clip_prior(x, inv_mean, inv_std):
    x_unorm = (x - inv_mean)/ inv_std
    dist_clip = torch.sum(torch.mean(torch.square(x_unorm - torch.clamp(x_unorm, 0.0, 1.0)), dim=0))
    return dist_clip


def order_invariant_prior(inputs, reorder_prior, epochs, device):
    epoch_size = len(inputs) // epochs

    x = torch.arange(epochs)
    y = torch.arange(epochs)
    xv, yv = torch.meshgrid(x, y)
    xv, yv = xv.reshape(-1), yv.reshape(-1)

    yv = torch.tile(yv, (epoch_size, )).reshape(epoch_size, -1).T * epoch_size + torch.arange(epoch_size)
    xv = torch.tile(xv, (epoch_size, )).reshape(epoch_size, -1).T * epoch_size + torch.arange(epoch_size)

    inputs_proj = inputs
    if reorder_prior.endswith('conv'):
        rand_conv = nn.Conv2d(inputs.shape[1], 96, kernel_size=3).to(device)
        inputs_proj = rand_conv(inputs)

    if reorder_prior.startswith('l2_mean'):
        inv_prior = invariant_prior_l2_mean(xv, yv, inputs_proj)
    elif reorder_prior.startswith('l2_max'):
        inv_prior = invariant_prior_l2_max(xv, yv, inputs_proj)
    else:
        raise TypeError(f"Unexpected reorder prior {reorder_prior}")

    return inv_prior


def invariant_prior_l2_mean(idx1, idx2, inputs):
    x1 = inputs[idx1]
    x2 = inputs[idx2]
    x1 = x1.mean(dim=0)
    x2 = x2.mean(dim=0)
    error = torch.mean(torch.square(x2 - x1))
    return error


def invariant_prior_l2_max(idx1, idx2, inputs):
    x1 = inputs[idx1]
    x2 = inputs[idx2]
    x1 = x1.amax(dim=1)
    x2 = x2.amax(dim=1)
    error = torch.mean(torch.square(x2 - x1))
    return error


def get_dummy_grads(model, features, labels, epochs, lr, k_batches, batch_size, criterion):
    data_len = len(features)
    assert data_len == k_batches * batch_size

    # the same method used in inverting gradients
    meta_model = MetaModel(model)
    meta_model_origin = copy.deepcopy(meta_model)

    model.train()
    for _ in range(epochs):
        # batch training
        for i in range(k_batches):
            # prepare batch data
            st_idx = i * batch_size
            en_idx = min((i + 1) * batch_size, data_len)
            dummy_x_batch = features[st_idx:en_idx]
            dummy_y_batch = labels[st_idx:en_idx]
            # forward pass
            dummy_pred_batch = meta_model(dummy_x_batch, meta_model.parameters)
            dummy_loss = criterion(dummy_pred_batch, dummy_y_batch)
            # gradients are calculated upon meta parameters
            dummy_grads = torch.autograd.grad(dummy_loss, meta_model.parameters.values(), create_graph=True)

            # this method can effectively transfer the grad_fn
            meta_model.parameters = OrderedDict(
                (n, p - lr * g)
                for (n, p), g in zip(meta_model.parameters.items(), dummy_grads)
            )

    # get the gradients for multiple steps
    meta_model.parameters = OrderedDict(
        (n, (p_o - p) / lr)
        for (n, p), p_o
        in zip(meta_model.parameters.items(), meta_model_origin.parameters.values())
    )
    return list(meta_model.parameters.values())