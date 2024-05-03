import torch
import torch.nn as nn
import torch.nn.functional as F

from fleak.attack.dummy import TorchDummy


def dlg(model, grads: list, dummy: TorchDummy, epochs: int, device="cpu"):
    """ Deep Leakage Gradient

    https://proceedings.neurips.cc/paper/2019/file/60a6c4002cc7b29142def8871531281a-Paper.pdf

    :param model: dlg model
    :param grads: gradients of the ground truth data
    :param dummy: TorchDummy object
    :param epochs: Number of epochs
    :param device: cpu or cuda
    :return: dummy data
    """
    model.eval()

    dummy_data = dummy.generate_dummy_input(device)
    dummy_label = dummy.generate_dummy_label(device)
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])  # default lr=1.0

    for iters in range(epochs):
        def closure():
            optimizer.zero_grad()

            dummy_pred = model(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = torch.mean(torch.sum(-dummy_onehot_label * F.log_softmax(dummy_pred, dim=-1), 1))
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            for dummy_g, origin_g in zip(dummy_dy_dx, grads):
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

    dummy_data = dummy.generate_dummy_input(device)
    # extract ground-truth labels proposed by iDLG
    label_pred = torch.argmin(torch.sum(grads[-2], dim=-1), dim=-1).detach().reshape((1,))

    optimizer = torch.optim.LBFGS([dummy_data], lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)

    for iters in range(epochs):
        def closure():
            optimizer.zero_grad()

            dummy_pred = model(dummy_data)
            dummy_loss = criterion(dummy_pred, label_pred)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            for dummy_g, origin_g in zip(dummy_dy_dx, grads):
                grad_diff += ((dummy_g - origin_g) ** 2).sum()
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)

    # save the dummy data
    dummy.append(dummy_data.detach())
    # save the label prediction
    dummy.append_label(label_pred)

    return dummy_data, label_pred
