import os
# os.chdir("../..")
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else device = "CPU"
# iteration = 300
# lr = 1.0


def reconstruct_dlg(shared_gradients, data, label, model, epochs=200, lr=1.0):
    dummy_data = data
    dummy_label = label
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
    # gradient closure
    for _ in range(epochs):
        def closure():
            optimizer.zero_grad()
            dummy_pred = model(dummy_data)
            dummy_label_onehot = F.softmax(dummy_label, dim=-1)
            dummy_loss = torch.mean(torch.sum(- dummy_label_onehot * F.log_softmax(dummy_pred, dim=-1), 1))
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            for dummy_g, origin_g in zip(dummy_dy_dx, shared_gradients):
                grad_diff += ((dummy_g - origin_g) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

    return dummy_data, dummy_label


def reconstruction_idlg(shared_gradients, data, label, model):
    dummy_data = data
    dummy_label = label
    optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)

    return dummy_data, dummy_label