"""GRNN: Generative Regression Neural Network—A Data Leakage Attack for Federated Learning

https://dl.acm.org/doi/abs/10.1145/3510032

"""

import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

from fleak.model.gan import GRNNGenerator


def grnn(model, gt_grads, dummy, rec_epochs=1000, rec_lr=0.0001, tv=1e-3, device="cpu"):
    """Official implementation of GRNN https://github.com/Rand2AI/GRNN

    Instead of directly recovering dummy data, GRNN optimizes the parameters of a Generator
    Dummy data are generated from random noise through a Generator
    Loss function is constructed by DLG loss + WD loss + TV loss
    RMSprop is utilized as the reconstruction optimizer

    :param model: inferred model
    :param gt_grads: gradients of the ground-truth data
    :param dummy: TorchDummy object
    :param rec_epochs: number of training epochs for the Generator
    :param rec_lr: reconstruct learning rate
    :param tv: hyperparameter for TV loss
    :param device: cpu or cuda
    :return: dummy data & dummy label
    """
    model.eval()

    flatten_gt_grads = torch.cat([g.view(-1) for g in gt_grads])

    generator = GRNNGenerator(dummy.n_classes, in_features=128, image_shape=dummy.image_shape).to(device)
    # update parameters of the generator other than dummy data
    optimizer = torch.optim.RMSprop(generator.parameters(), lr=rec_lr, momentum=0.99)
    tv_loss = TVLoss()
    random_noise = torch.randn(dummy.batch_size, 128).to(device)

    iter_bar = tqdm(range(rec_epochs),
                    total=rec_epochs,
                    desc=f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}')
    for _ in iter_bar:
        optimizer.zero_grad()

        # produce dummy data & label
        # softmax already produced upon dummy label
        dummy_data, dummy_label = generator(random_noise)
        dummy_pred = model(dummy_data)
        dummy_loss = - torch.mean(torch.sum(dummy_label * torch.log(torch.softmax(dummy_pred, 1)), dim=-1))
        # obtain fake gradient
        dummy_grads = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

        flatten_dummy_grads = torch.cat([g.view(-1) for g in dummy_grads])
        grad_diff_l2 = ((flatten_dummy_grads - flatten_gt_grads) ** 2).sum()
        grad_diff_wd = wasserstein_distance(flatten_dummy_grads, flatten_gt_grads)
        # total variation loss
        # tv = 1e-3 for LeNet, and tv = 1e-6 for ResNet18
        loss_tv = tv * tv_loss(dummy_data)
        grad_diff = grad_diff_l2 + grad_diff_wd + loss_tv

        grad_diff.backward()
        optimizer.step()
        iter_bar.set_postfix(loss_l2=np.round(grad_diff_l2.item(), 8),
                             loss_wd=np.round(grad_diff_wd.item(), 8),
                             loss_tv=np.round(loss_tv.item(), 8))

    dummy.append(dummy_data)
    rec_dummy_label = torch.argmax(dummy_label, dim=-1)
    dummy.append_label(rec_dummy_label)

    return dummy_data, rec_dummy_label


def wasserstein_distance(first_samples, second_samples, p=2):
    """wasserstein distance

    The calculation method is copied from the author of GRNN,
    However, it seems incorrect ?

    """
    w = torch.abs(first_samples[0] - second_samples[0])
    w = torch.pow(torch.sum(torch.pow(w, p)), 1. / p)
    return torch.pow(torch.pow(w, p).mean(), 1. / p)


class TVLoss(nn.Module):
    """Total variation loss

    This implementation is provided by the author of GRNN

    """

    def __init__(self, weight=1):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def _tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
