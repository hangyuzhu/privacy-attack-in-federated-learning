import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from ..server.dummy import TorchDummyImage


def criterion(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def generate_dummy_k(dummy, device):
    """ Generate dummy data with Kaiming initialization

     This may be helpful for stable generation

     """
    dummy_data = torch.empty(dummy.input_shape).to(device).requires_grad_(True)
    # equivalent to the default initialization of pytorch
    nn.init.kaiming_uniform_(dummy_data, a=math.sqrt(5))
    dummy_label = torch.empty(dummy.label_shape).to(device).requires_grad_(True)
    nn.init.kaiming_uniform_(dummy_label, a=math.sqrt(5))
    return dummy_data, dummy_label


def generate_dummy(dummy, device):
    dummy_data = torch.randn(dummy.input_shape).to(device).requires_grad_(True)
    dummy_label = torch.randn(dummy.label_shape).to(device).requires_grad_(True)
    return dummy_data, dummy_label


def dlg(model, grads: OrderedDict, dummy: TorchDummyImage, epochs: int, device="cpu"):
    """ Deep Leakage Gradient

    https://proceedings.neurips.cc/paper/2019/file/60a6c4002cc7b29142def8871531281a-Paper.pdf

    :param model: dlg model
    :param grads: gradients of the ground truth data
    :param dummy: TorchDummyImage object
    :param epochs: Number of epochs
    :param device: cpu or cuda
    :return: dummy data
    """
    model.eval()

    dummy_data, dummy_label = generate_dummy(dummy, device)
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])  # default lr=1.0

    for iters in range(epochs):
        def closure():
            optimizer.zero_grad()

            dummy_pred = model(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            for dummy_g, origin_g in zip(dummy_dy_dx, grads.values()):
                grad_diff += ((dummy_g - origin_g)**2).sum()
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)

    # save the dummy data
    dummy.append(dummy_data.detach())
    # save the dummy label
    dummy.append_label(torch.argmax(dummy_label, dim=-1).item())

    return dummy_data, dummy_label


def idlg(model, grads, dummy, epochs=300, lr=0.075, device="cpu"):
    """Improved Deep Leakage Gradients

    iDLG theoretically gives label prediction
    https://arxiv.org/pdf/2001.02610.pdf

    :param model: idlg model
    :param grads: gradients of the ground truth data
    :param dummy: TorchDummy object
    :param epochs: number of epochs
    :param lr: learning rate
    :param device: cpu or cuda
    :return: dummy data, label prediction
    """
    model.eval()

    dummy_data, dummy_label = generate_dummy(dummy, device)
    optimizer = torch.optim.LBFGS([dummy_data], lr=lr)

    # extract ground-truth labels proposed by iDLG
    label_pred = torch.argmin(torch.sum(list(grads.values())[-2], dim=-1), dim=-1).detach().reshape((1,))
    idlg_criterion = nn.CrossEntropyLoss().to(device)

    for iters in range(epochs):
        def closure():
            optimizer.zero_grad()

            dummy_pred = model(dummy_data)
            dummy_loss = idlg_criterion(dummy_pred, label_pred)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            for dummy_g, origin_g in zip(dummy_dy_dx, grads.values()):
                grad_diff += ((dummy_g - origin_g) ** 2).sum()
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)

    # save the dummy data
    dummy.append(dummy_data.detach())
    # save the label prediction
    dummy.append_label(label_pred)

    return dummy_data, label_pred
