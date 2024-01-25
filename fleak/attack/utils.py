import torch
import os
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
import PIL.Image as Image
import torch.nn as nn
from torch.nn.modules.utils import _pair, _quadruple

def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections /= torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections

def wasserstein_distance(first_samples,
                         second_samples,
                         p=2,
                         device='cuda'):
    wasserstein_distance = torch.abs(first_samples[0] - second_samples[0])
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p)), 1. / p)
    return torch.pow(torch.pow(wasserstein_distance, p).mean(), 1. / p)


def flatten_gradients(dy_dx):
    flatten_dy_dx = None
    for layer_g in dy_dx:
        if flatten_dy_dx is None:
            flatten_dy_dx = torch.flatten(layer_g)
        else:
            flatten_dy_dx = torch.cat((flatten_dy_dx, torch.flatten(layer_g)))
    return flatten_dy_dx

def gen_dataset(dataset, data_path, shape_img):
    class Dataset_from_Image(Dataset):
        def __init__(self, imgs, labs, transform=None):
            self.imgs = imgs
            self.labs = labs
            self.transform = transform
            del imgs, labs

        def __len__(self):
            return self.labs.shape[0]

        def __getitem__(self, idx):
            lab = self.labs[idx]
            img = Image.open(self.imgs[idx])
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = self.transform(img)
            return img, lab

    def face_dataset(path, num_classes):
        images_all = []
        index_all = []
        folders = os.listdir(path)
        for foldidx, fold in enumerate(folders):
            if foldidx+1==num_classes: break
            if os.path.isdir(os.path.join(path, fold)):
                files = os.listdir(os.path.join(path, fold))
                for f in files:
                    if len(f) > 4:
                        images_all.append(os.path.join(path, fold, f))
                        index_all.append(foldidx)
        transform = transforms.Compose([transforms.Resize(shape_img),
                                 transforms.ToTensor()])
        dst = Dataset_from_Image(images_all, np.asarray(index_all, dtype=int), transform=transform)
        return dst
    if dataset == 'mnist':
        num_classes = 10
        tt = transforms.Compose([transforms.Resize(shape_img),
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor()
                                 ])
        dst = datasets.MNIST(os.path.join(data_path, 'mnist/'), download=True, transform=tt)
    elif dataset == 'cifar100':
        num_classes = 100
        tt = transforms.Compose([transforms.Resize(shape_img),
                                 transforms.ToTensor()])
        dst = datasets.CIFAR100(os.path.join(data_path, 'cifar100/'), download=True, transform=tt)
    elif dataset == 'lfw':
        num_classes = 5749
        dst = face_dataset(os.path.join(data_path, 'lfw/'), shape_img)
    elif dataset == 'VGGFace':
        num_classes = 2622
        dst = face_dataset(os.path.join(data_path, 'VGGFace/vgg_face_dataset/'), num_classes)
    else:
        exit('unknown dataset')
    return dst, num_classes

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    @staticmethod
    def _tensor_size(t):
        return t.size()[1]*t.size()[2]*t.size()[3]


def loss_f(loss_name, flatten_fake_g, flatten_true_g, device):
    if loss_name == 'l2':
        grad_diff = ((flatten_fake_g - flatten_true_g) ** 2).sum()
        # grad_diff = torch.sqrt(((flatten_fake_g - flatten_true_g) ** 2).sum())
    elif loss_name == 'wd':
        grad_diff = wasserstein_distance(flatten_fake_g.view(1, -1), flatten_true_g.view(1, -1),
                                         device=f'cuda:{device}')
    else:
        raise Exception('Wrong loss name.')
    return grad_diff


class MedianPool2d(nn.Module):
    """Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=True):
        """Initialize with kernel_size, stride, padding."""
        super().__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x