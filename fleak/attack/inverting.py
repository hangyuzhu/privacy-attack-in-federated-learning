import torch
from .inverting_class import GradientReconstructor,FedAvgReconstructor


def ig(local_model, local_gradients, device="cpu"):
    """
    Inverting Gradients: https://proceedings.neurips.cc/paper/2020/file/c4ede56bbd98819ae6112b20ac6bf145-Paper.pdf
    :param local_model: uploaded local model
    :param local_gradients: global model - local model
    :param device: cpu or cuda
    :return: reconstructed data and labels
    """
    # dm = torch.as_tensor([0.4915, 0.4823, 0.4468], device=device, dtype=torch.float32)[None, :, None, None]
    # ds = torch.as_tensor([0.2470, 0.2435, 0.2616], device=device, dtype=torch.float32)[None, :, None, None]
    dm = torch.as_tensor([0.4914, 0.4822, 0.4465], device=device, dtype=torch.float32)[None, :, None, None]
    ds = torch.as_tensor([0.2023, 0.1994, 0.2010], device=device, dtype=torch.float32)[None, :, None, None]

    local_model.zero_grad()
    # --------------------------------------------------
    # generate_data = dummy_data.detach()
    # --------------------------------------------------
    label_pred = torch.argmin(torch.sum(list(local_gradients.values())[-2], dim=-1), dim=-1).detach().reshape((1,))
    config = dict(signed=True,
                  boxed=True,
                  cost_fn='sim',
                  indices='def',
                  weights='equal',
                  lr=0.1,
                  optim='adam',
                  restarts=1,
                  max_iterations=4000,
                  total_variation=1e-6,
                  init='randn',
                  filter='none',
                  lr_decay=True,
                  scoring_choice='loss')
    rec_machine = GradientReconstructor(local_model, (dm, ds), config, num_images=1, device=device)
    output, stats = rec_machine.reconstruct(local_gradients, label_pred)
    return output, label_pred


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
    return output, stats