""" Inverting Gradients

 https://proceedings.neurips.cc/paper/2020/file/c4ede56bbd98819ae6112b20ac6bf145-Paper.pdf

 """
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from .inverting_class import FedAvgReconstructor


def Single(model, grads, dummy, epochs=4000, lr=0.1, alpha=1e-6, device="cpu"):
    """ Recover a single image from single gradients

    Similar to DLG based methods but adopting cosine similarity as the loss function

    :param model: inferred model
    :param grads: gradients of the ground truth data
    :param device: cpu or cuda
    :return: reconstructed data and labels
    """
    model.eval()

    dummy_data = dummy.generate_dummy_input(device)
    label_pred = torch.argmin(torch.sum(list(grads.values())[-2], dim=-1), dim=-1).detach().reshape((1,))

    optimizer = optim.Adam([dummy_data], lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[epochs // 2.667, epochs // 1.6, epochs // 1.142],
        gamma=0.1
    )  # 3/8 5/8 7/8
    criterion = nn.CrossEntropyLoss().to(device)

    for i in range(epochs):
        def closure():
            optimizer.zero_grad()
            model.zero_grad()

            dummy_pred = model(dummy_data)
            dummy_loss = criterion(dummy_pred, label_pred)
            dummy_grads = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            rec_loss = cosine_similarity_loss(dummy_grads, grads)
            rec_loss += alpha * total_variation(dummy_data)
            rec_loss.backward()

            # small trick 1: convert the grad to 1 or -1
            dummy_data.grad.sign_()
            return rec_loss

        rec_loss = optimizer.step(closure)
        scheduler.step()

        with torch.no_grad():
            # small trick 2: project into image space
            dummy_data.data = torch.max(
                torch.min(dummy_data, (1 - dummy.t_dm) / dummy.t_ds), -dummy.t_dm / dummy.t_ds)
            if i + 1 == epochs:
                print(f'Epoch: {i + 1}. Rec. loss: {rec_loss.item():2.4f}.')

    dummy.append(dummy_data.detach())
    dummy.append_label(label_pred)

    return dummy_data, label_pred


def cosine_similarity_loss(dummy_grads: tuple, grads: OrderedDict):
    """ Compute cosine similarity value

    Compared to L2-norm loss, it can additionally capture the direction information

    :param dummy_grads: gradients of dummy data
    :param grads: gradients of the ground truth data
    :return: the loss value
    """
    # numerator
    nu = 0
    # denominator
    dn0 = 0
    dn1 = 0
    for dg, g in zip(dummy_grads, grads.values()):
        # equivalent to the inner product of two vectors
        nu += (dg * g).sum()
        dn0 += dg.pow(2).sum()
        dn1 += g.pow(2).sum()
    loss = 1 - nu / dn0.sqrt() / dn1.sqrt()  # l2-norm
    return loss


def total_variation(x):
    """ Total variation

    https://cbio.mines-paristech.fr/~jvert/svn/bibli/local/Rudin1992Nonlinear.pdf

     """
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def ig_weight(local_model, local_gradients, device="cpu"):
    dm = torch.as_tensor([0.4914, 0.4822, 0.4465], device=device, dtype=torch.float32)[None, :, None, None]
    ds = torch.as_tensor([0.2023, 0.1994, 0.2010], device=device, dtype=torch.float32)[None, :, None, None]
    label_pred = torch.argmin(torch.sum(list(local_gradients.values())[-2], dim=-1), dim=-1).detach().reshape((1,))
    local_model.zero_grad()
    config = dict(signed=True,
                  boxed=True,
                  cost_fn='sim',
                  indices='def',
                  weights='equal',
                  lr=0.1,
                  optim='adam',
                  restarts=1,
                  max_iterations=8000,
                  total_variation=1e-6,
                  init='randn',
                  filter='none',
                  lr_decay=True,
                  scoring_choice='loss')
    rec_machine = FedAvgReconstructor(local_model, (dm, ds), local_steps=5, local_lr=1e-4, config=config, use_updates=True,device=device)
    output, stats = rec_machine.reconstruct(local_gradients, label_pred, img_shape=(3, 32, 32))
    return output, label_pred


def ig_multiple(local_model, local_gradients, device="cpu"):
    dm = torch.as_tensor([0.4914, 0.4822, 0.4465], device=device, dtype=torch.float32)[None, :, None, None]
    ds = torch.as_tensor([0.2023, 0.1994, 0.2010], device=device, dtype=torch.float32)[None, :, None, None]
    local_steps = 5
    local_lr = 1e-4
    use_updates = True
    labels_pred = torch.argmin(torch.sum(list(local_gradients.values())[-2], dim=-1), dim=-1).detach().reshape((1,))
    local_model.zero_grad()
    config = dict(signed=True,
                  boxed=True,
                  cost_fn='sim',
                  indices='def',
                  weights='equal',
                  lr=1,
                  optim='adam',
                  restarts=8,
                  max_iterations=24000,
                  total_variation=1e-6,
                  init='randn',
                  filter='none',
                  lr_decay=True,
                  scoring_choice='loss')
    rec_machine = FedAvgReconstructor(local_model, (dm, ds), local_steps, local_lr, config, use_updates=use_updates, num_images=8)
    output, stats = rec_machine.reconstruct(local_gradients, labels_pred, img_shape=(3, 32, 32))
    return output, labels_pred