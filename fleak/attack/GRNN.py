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
setup = dict(device="cuda" if torch.cuda.is_available() else "CPU", dtype=torch.float32)
dm = torch.as_tensor([0.4914672374725342, 0.4822617471218109, 0.4467701315879822], device="cuda" if torch.cuda.is_available() else "CPU", dtype=torch.float32)[None, :, None, None]
ds = torch.as_tensor([0.24703224003314972, 0.24348513782024384, 0.26158785820007324], device="cuda" if torch.cuda.is_available() else "CPU", dtype=torch.float32)[None, :, None, None]



def Reconstruction(num_class, global_model, shape_img, data_loader):
    batchsize = 1
    iteration = 1000
    num_exp = 10
    g_in = 128
    plot_num = 100
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
        # flatten_true_g = flatten_gradients(shared_gradients.values())
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
                # history_l.append(fake_label)
                # history.append(fake_out)
                history.append([tp(fake_out[imidx].detach().cpu()) for imidx in range(batchsize)])
                history_l.append([fake_label.argmax(dim=1)[imidx].item() for imidx in range(batchsize)])
            torch.cuda.empty_cache()
            del generator_loss, generate_dy_dx, flatten_fake_g, grad_diff_l2, grad_diff_wd, grad_diff, tvloss
        ## show reconstructions
        for imidx in range(batchsize):
            plt.figure(figsize=(12, 8))
            plt.subplot(plot_num // 10, 10, 1)
            plt.imshow(tp(gt_data[imidx].cpu()))
            for i in range(min(len(history), plot_num - 1)):
                plt.subplot(plot_num // 10, 10, i + 2)
                plt.imshow(history[i][imidx])
                plt.title('l=%d' % (history_l[i][imidx]))
                # plt.title('i=%d,l=%d' % (history_iters[i], history_l[i][imidx]))
                plt.axis('off')
            path = r'D:\leakage-attack-in-federated-learning\saved_results'
            if not os.path.exists(path):
                os.makedirs(path)
            history[-1][imidx].save(os.path.join(path, f'GRNN_fake_image.png'))
            plt.close()
            # for i, _recon in enumerate(history):
            #     _recon.mul_(ds).add_(dm).clamp_(min=0, max=1)
            #     _recon = _recon.to(dtype=torch.float32)
            #     plt.subplot(10, 20, i + 1)
            #     plt.imshow(_recon[0].permute(1, 2, 0).cpu())
            #     plt.axis('off')
            # path = r'D:\leakage-attack-in-federated-learning\saved_results'
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # plt.savefig(os.path.join(path, 'GRNN' + '_fake_image.png'))
        torch.cuda.empty_cache()
        history.clear()
        history_l.clear()
        iter_bar.close()
