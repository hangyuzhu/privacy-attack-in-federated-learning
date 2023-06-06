import os
os.chdir("..")
import breaching
import torch
import logging, sys
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()
'''
import numpy as np
from pprint import pprint
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms

from utils import label_to_onehot
from utils import cross_entropy_for_onehot
from model import LeNet

def weight_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5,0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5,0.5)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

dst = datasets.CIFAR100("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

img_index = 10
gt_data = tp(dst[img_index][0]).to(device)

gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
gt_label = gt_label.view(1,)
gt_onehot_label = label_to_onehot(gt_label)

#LeNet
net = LeNet().to(device)

torch.manual_seed(42)
#weight init
net.apply(weight_init)

criterion = nn.CrossEntropyLoss().to(device)

#Number of iterations (Write only a multiple of 100)
iteration = 300

#Predicted value of example image
pred = net(gt_data)
print(pred)
y = criterion(pred, gt_onehot_label)
print(y)
#Take original dy_dx
dy_dx = torch.autograd.grad(y, net.parameters())
original_dy_dx = list((_.detach().clone() for _ in dy_dx))

original_dy_dx = list((_.detach().clone() for _ in dy_dx))


#Create Dummy image
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)

# without dummy_label
optimizer = torch.optim.LBFGS([dummy_data, ])
# Get Ground-truth label
label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)

criterion = nn.CrossEntropyLoss().to(device)

history = []
for iters in range(iteration):
    def closure():
        optimizer.zero_grad()

        dummy_pred = net(dummy_data)
        dummy_loss = criterion(dummy_pred, label_pred)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()

        return grad_diff


    optimizer.step(closure)

    if iters % 10 == 0:
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        history.append(tt(dummy_data[0].cpu()))

plt.figure(figsize=(12, 8))
for i in range(int(iteration / 10)):
    plt.subplot(int(iteration / 100), 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')

plt.show()
'''
# initialize
cfg = breaching.get_config(overrides=["case=1_single_image_small", "attack=deepleakage"])

device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))

cfg.case.data.partition="unique-class"
cfg.case.user.user_idx = 1

cfg.case.user.provide_labels = False
# This attack can reconstruct label information via optimization.

user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
breaching.utils.overview(server, user, attacker)

server_payload = server.distribute_payload()
shared_data, true_user_data = user.compute_local_updates(server_payload)

reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], {}, dryrun=cfg.dryrun)

# metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, [server_payload],
#                                     server.model, order_batch=True, compute_full_iip=False,
#                                     cfg_case=cfg.case, setup=setup)
user.plot(reconstructed_user_data, scale=True)
