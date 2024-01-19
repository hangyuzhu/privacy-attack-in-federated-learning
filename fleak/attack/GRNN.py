import time, datetime


from .utils import *
from fleak.model.grnn_network import Generator
import torch
import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms

from tqdm import tqdm
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "CPU"


def Reconstruction(num_class, global_model, shape_img, data_loader):
    batchsize = 1
    iteration = 1000
    num_exp = 10
    g_in = 128
    plot_num = 30
    tp = transforms.Compose([transforms.ToPILImage()])
    criterion = nn.CrossEntropyLoss()
    for idx_net in range(num_exp):
        generator = Generator(num_class, channel=3, shape_img=shape_img[0], batchsize=batchsize, g_in=g_in).to(device)
        generator.weight_init(mean=0.0, std=0.02)
        G_optimizer =torch.optim.RMSprop(generator.parameters(), lr=0.0001, momentum=0.99)
        tv_loss = TVLoss()
        gt_data, gt_label = next(iter(data_loader))
        gt_data, gt_label = gt_data.to(device), gt_label.to(device)
        pred = global_model(gt_data)
        y = criterion(pred, gt_label)
        dy_dx = torch.autograd.grad(y, global_model.parameters())
        flatten_true_g = flatten_gradients(dy_dx)
        flatten_true_g = flatten_true_g.to(device)
        ##  initialize the random input
        random_noise = torch.randn(batchsize, g_in)
        random_noise = random_noise.to(device)
        iter_bar = tqdm(range(iteration),
                        total=iteration,
                        desc=f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}',
                        ncols=180)
        history = []
        history_l = []


        ## GRNN train ##
        for iters in iter_bar:
            fake_out, fake_label = generator(random_noise)
            fake_out, fake_label = fake_out.to(device), fake_label.to(device)
            fake_pred = global_model(fake_out)
            generator_loss = - torch.mean(torch.sum(fake_label * torch.log(torch.softmax(fake_pred, 1)), dim=-1))
            generate_dy_dx = torch.autograd.grad(generator_loss, global_model.parameters(), create_graph=True)
            flatten_fake_g = flatten_gradients(generate_dy_dx)
            flatten_fake_g = flatten_fake_g.to(device)
            grad_diff_l2 = loss_f('l2', flatten_fake_g, flatten_true_g, 0)
            grad_diff_wd = loss_f('wd', flatten_fake_g, flatten_true_g, 0)
            tvloss = 1e-6 * tv_loss(fake_out)
            grad_diff = grad_diff_l2 + grad_diff_wd + tvloss  # loss for GRNN
            G_optimizer.zero_grad()
            grad_diff.backward()
            G_optimizer.step()
            iter_bar.set_postfix(loss_l2=np.round(grad_diff_l2.item(), 8),
                                 loss_wd=np.round(grad_diff_wd.item(), 8),
                                 loss_tv=np.round(tvloss.item(), 8),
                                 img_mses=round(torch.mean(abs(fake_out - gt_data)).item(), 8))
            if iters % int(iteration / plot_num) == 0:
                history.append([tp(fake_out[imidx].detach().cpu()) for imidx in range(batchsize)])
                history_l.append([fake_label.argmax(dim=1)[imidx].item() for imidx in range(batchsize)])
            torch.cuda.empty_cache()
            del generator_loss, generate_dy_dx, flatten_fake_g, grad_diff_l2, grad_diff_wd, grad_diff, tvloss
        for imidx in range(batchsize):
            plt.figure(figsize=(12, 8))
            plt.subplot(plot_num // 10, 10, 1)
            plt.imshow(tp(gt_data[imidx].cpu()))
            for i in range(min(len(history), plot_num-1)):
                plt.subplot(plot_num//10, 10, i + 2)
                plt.imshow(history[i][imidx])
                plt.title('l=%d' % (history_l[i][imidx]))
                # plt.title('i=%d,l=%d' % (history_iters[i], history_l[i][imidx]))
                plt.axis('off')

    return history, history_l