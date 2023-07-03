import os
# os.chdir("../..")
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy

device = "cuda" if torch.cuda.is_available() else "CPU"
# criterion = torch.nn.CrossEntropyLoss().to(device)
def criterion(pred,target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def reconstruct_dlg(shared_gradients, dummy_data, dummy_label, model, epochs=300, lr=0.075):

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
    # gradient closure
    minimal_value_so_far = torch.as_tensor(float("inf"), device="cuda" if torch.cuda.is_available() else "CPU", dtype=torch.float32)
    for iters in range(epochs):
        global best_dummy_data,best_dummy_label
        def closure():
            optimizer.zero_grad()
            dummy_pred = model(dummy_data)
            dummy_label_onehot = F.softmax(dummy_label, dim=-1)
            dummy_loss = torch.mean(torch.sum(- dummy_label_onehot * F.log_softmax(dummy_pred, dim=-1), 1))
            dummy_loss = criterion(dummy_pred,dummy_label_onehot)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            for dummy_g, origin_g in zip(dummy_dy_dx, shared_gradients.values()):
                grad_diff += ((dummy_g - origin_g) ** 2).sum()
            grad_diff.backward()

            return grad_diff

        objective_value = optimizer.step(closure)
        with torch.no_grad():
            if objective_value < minimal_value_so_far:
                minimal_value_so_far = objective_value.detach()
                best_dummy_data = dummy_data.detach().clone()
                best_dummy_label = dummy_label.detach().clone()

        if iters % 10 == 0:
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())


    return dummy_data,dummy_label


def reconstruct_idlg(shared_gradients, dummy_data, dummy_label, model, epochs=200, lr=0.075):
    origin_gradients = shared_gradients.copy()
    optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
    # label_pred = torch.argmin(torch.sum(shared_gradients, dim=-1), dim=-1).detach().reshape((1,)).requires_grad(False)
    ##  use the gradients of FC1 to recover the ground-truth label(but has an error: TypeError: 'bool' object is not callable)
    label_pred = torch.argmin(torch.sum(list(origin_gradients.values())[-4])).detach().reshape((1,))
    criterion_idlg = nn.CrossEntropyLoss().to(device)
    for iters in range(epochs):
        def closure():
            optimizer.zero_grad()
            dummy_pred = model(dummy_data)
            dummy_loss = criterion_idlg(dummy_pred, label_pred)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
            grad_diff = 0
            for dummy_g, origin_g in zip(dummy_dy_dx, shared_gradients.values()):
                grad_diff += ((dummy_g - origin_g) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

        if iters % 10 == 0:
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())

    return dummy_data, dummy_label