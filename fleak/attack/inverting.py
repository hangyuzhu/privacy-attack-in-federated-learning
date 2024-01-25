import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from .inverting_class import GradientReconstructor

device = "cuda" if torch.cuda.is_available() else "CPU"

dm = torch.as_tensor([0.4915, 0.4823, 0.4468], device="cuda" if torch.cuda.is_available() else "CPU", dtype=torch.float32)[None, :, None, None]
ds = torch.as_tensor([0.2470, 0.2435, 0.2616], device="cuda" if torch.cuda.is_available() else "CPU", dtype=torch.float32)[None, :, None, None]


def recontstruction(global_model, shared_gradient,dummy_data):
    global_model.zero_grad()
    generate_data = dummy_data.detach()
    label_pred = torch.argmin(torch.sum(list(shared_gradient.values())[-4])).detach().reshape((1,))
    config = dict(signed=True,
                  boxed=True,
                  cost_fn='sim',
                  indices='def',
                  weights='equal',
                  lr=0.1,
                  optim='adam',
                  restarts=32,
                  max_iterations=1000,
                  total_variation=1e-4,
                  init='randn',
                  filter='none',
                  lr_decay=True,
                  scoring_choice='loss')
    rec_machine = GradientReconstructor(global_model, (dm, ds), config, num_images=1)
    output, stats = rec_machine.reconstruct(shared_gradient, label_pred,  x=generate_data)
    return output, label_pred