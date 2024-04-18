import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict

from ..server.dummy import TorchDummy


def criterion(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def generate_dummy(dummy: TorchDummy, device: str):
    import math
    dummy_data = torch.empty(dummy.data_shape).to(device).requires_grad_(True)
    nn.init.kaiming_uniform_(dummy_data, a=math.sqrt(5))
    # dummy_data = torch.randn(dummy.data_shape).to(device).requires_grad_(True)
    dummy_label = torch.randn(dummy.label_shape).to(device).requires_grad_(True)
    return dummy_data, dummy_label


def dlg(model, grads: OrderedDict, dummy: TorchDummy, epochs: int, device="cpu"):
    """ Deep Leakage Gradient

    https://proceedings.neurips.cc/paper/2019/file/60a6c4002cc7b29142def8871531281a-Paper.pdf

    :param model: dlg model
    :param grads: model gradients of the ground truth data
    :param dummy: TorchDummy object
    :param epochs: Number of epochs
    :param device: cpu or cuda
    :return: dummy data
    """
    model.eval()

    dummy_data, dummy_label = generate_dummy(dummy, device)
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    best_score = torch.inf
    best_dummy_data = dummy_data

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

        # select the dummy data with the minimum loss value
        loss = optimizer.step(closure)
        if loss.item() < best_score:
            best_score = loss.item()
            best_dummy_data = dummy_data.detach().clone()

    # save the history
    dummy.append(best_dummy_data)

    return best_dummy_data, dummy_label


def idlg(global_model, local_grads, dummy_data, epochs=200, lr=0.075, device="cpu"):
    """
    https://arxiv.org/pdf/2001.02610.pdf
    :param global_model: global model
    :param local_grads: uploaded gradients
    :param dummy_data: fake data
    :param epochs: number of epochs
    :param lr: learning rate
    :param device: cpu or cuda
    :return: dummy data
    """
    global_model.eval()
    optimizer = torch.optim.Adam([dummy_data], lr=lr)
    minimal_value_so_far = torch.as_tensor(float("inf"), device=device, dtype=torch.float32)
    label_pred = torch.argmin(torch.sum(list(local_grads.values())[-2], dim=-1), dim=-1).detach().reshape((1,))
    criterion_idlg = nn.CrossEntropyLoss().to(device)

    for iters in range(epochs):
        def closure():
            optimizer.zero_grad()
            global_model.zero_grad()
            dummy_pred = global_model(dummy_data)
            dummy_loss = criterion_idlg(dummy_pred, label_pred)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, global_model.parameters(), create_graph=True)
            grad_diff = 0
            for dummy_g, origin_g in zip(dummy_dy_dx, local_grads.values()):
                grad_diff += ((dummy_g - origin_g) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        objective_value = optimizer.step(closure)
        with torch.no_grad():
            if objective_value < minimal_value_so_far:
                minimal_value_so_far = objective_value.detach()
                best_dummy_data = dummy_data.detach().clone()
    return best_dummy_data, label_pred


if __name__ == "__main__":
    x = torch.tensor(5.)
    x.requires_grad = True
    w = torch.tensor(3.)
    w.requires_grad = True
    # y = x**3

    z = x*w
    dw = torch.autograd.grad(z, w, retain_graph=True)
    # z.backward()
    # print(w.grad)
    l = (dw[0] - 1) ** 2
    l.backward()
    print(x.grad)

