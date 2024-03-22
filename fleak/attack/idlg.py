import os
# os.chdir("../..")
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy


def criterion(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def dlg(global_model, local_grads, dummy_data, dummy_label, epochs, lr=1.0, device="cpu"):
    """
    https://proceedings.neurips.cc/paper/2019/file/60a6c4002cc7b29142def8871531281a-Paper.pdf
    :param global_model: global model
    :param local_grads: uploaded gradients
    :param dummy_data: fake data
    :param dummy_label: fake labels
    :param epochs: number of epochs
    :param device: cpu or cuda
    :return: dummy data
    """
    global_model.eval()
    optimizer = torch.optim.LBFGS([dummy_data], lr=lr)

    # gradient closure
    minimal_value_so_far = torch.as_tensor(float("inf"), device=device, dtype=torch.float32)
    best_dummy_data = None
    for iters in range(epochs):
        def closure():
            optimizer.zero_grad()
            global_model.zero_grad()
            dummy_pred = global_model(dummy_data)
            dummy_loss = criterion(dummy_pred, dummy_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, global_model.parameters(), create_graph=True)

            grad_diff = 0
            for dummy_g, origin_g in zip(dummy_dy_dx, local_grads.values()):
                grad_diff += ((dummy_g - origin_g).pow(2)).sum()
            grad_diff.backward()
            return grad_diff

        objective_value = optimizer.step(closure)
        with torch.no_grad():
            if objective_value < minimal_value_so_far:
                minimal_value_so_far = objective_value.detach()
                best_dummy_data = dummy_data.detach().clone()
    return best_dummy_data


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

