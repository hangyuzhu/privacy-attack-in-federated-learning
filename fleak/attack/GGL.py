import torch
import torch.nn as nn
from fleak.attack.GGL_CLASS import GGL_reconstruction


device = "cuda" if torch.cuda.is_available() else "CPU"


def GGLreconstruction(global_model, generator, shared_gradients, budget = 500, use_tanh=True):
    loss_fn = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    ggl_reconstruct = GGL_reconstruction(fl_model=global_model, generator=generator, loss_fn=loss_fn, search_dim=(100,), budget=budget, use_tanh=use_tanh)
    x_res, dummy_label = ggl_reconstruct.reconstruct(shared_gradients)
    return x_res, dummy_label