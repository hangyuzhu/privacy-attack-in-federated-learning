import torch
import os
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
import PIL.Image as Image
import torch.nn as nn
from torch.nn.modules.utils import _pair, _quadruple
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
# from ..data.image_dataset import IMAGE_MEAN, IMAGE_STD


def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections /= torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections


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


def normalize_lpips(x):
    return (x - 0.5) * 2

def normalize_ds(inp, method=None, ds="imagenet"):
    device = inp.device
    if method is None:
        pass
    elif method == "ds":
        mean = torch.tensor(ds_mean[ds], device=device).view(1, 3, 1, 1)
        std = torch.tensor(ds_std[ds], device=device).view(1, 3, 1, 1)
        inp = torch.clamp((inp * std) + mean, 0, 1)
    else:
        raise ValueError(f"Unknown method {method}")
    return inp


class Eval:
    def __init__(self, fix_order_method=None, fix_sign_method=None, ds="cifar10", device="CPU"):
        # self.S = s
        self.device = device
        self.compute_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(
            self.device
        )
        self.eps = 1e-20
        self.fix_order_method = fix_order_method
        self.fix_sign_method = fix_sign_method
        # self.n = self.S.shape[0]
        self.ds = ds

    def fix_order(self, S_hat):
        assert S_hat.shape[0] == self.n

        if self.fix_order_method is None:
            return S_hat
        elif self.fix_order_method == "ssim":
            S_hat_inv = 1 - S_hat
            scores = []
            for S_i in self.S.unsqueeze(1):
                for S_hat_i, S_hat_inv_i in zip(
                    S_hat.unsqueeze(1), S_hat_inv.unsqueeze(1)
                ):
                    score = (
                        structural_similarity_index_measure(S_hat_i, S_i).cpu().item()
                    )
                    score_inv = (
                        structural_similarity_index_measure(S_hat_inv_i, S_i)
                        .cpu()
                        .item()
                    )
                    scores.append(np.max([score, score_inv]))
            similarity_matrix = torch.tensor(np.array(scores), device=self.device).view(
                self.n, self.n
            )
        elif self.fix_order_method == "lpips":
            S_hat_inv = 1 - S_hat
            scores = []
            for S_i in self.S.unsqueeze(1):
                for S_hat_i, S_hat_inv_i in zip(
                    S_hat.unsqueeze(1), S_hat_inv.unsqueeze(1)
                ):

                    score = (
                        self.compute_lpips(
                            normalize_lpips(S_hat_i), normalize_lpips(S_i)
                        )
                        .cpu()
                        .item()
                    )
                    score_inv = (
                        self.compute_lpips(
                            normalize_lpips(S_hat_inv_i), normalize_lpips(S_i)
                        )
                        .cpu()
                        .item()
                    )
                    scores.append(np.min([score, score_inv]))
            # invert because lower is better
            similarity_matrix = -torch.tensor(
                np.array(scores), device=self.device
            ).view(self.n, self.n)
        elif self.fix_order_method == "cs":
            # Note: We are matching based on absolute value of cosine similarity.
            # Ideally this should be done only for CPA (where sign is not preserved) and not for GMA
            S_unit = self.S / self.S.norm(dim=-1, keepdim=True)
            S_hat_unit = S_hat / S_hat.norm(dim=-1, keepdim=True)
            cs_matrix = torch.matmul(S_unit, S_hat_unit.T).abs()
            similarity_matrix = cs_matrix
        else:
            raise ValueError("Unknown method: ", self.fix_order_method)

        sorted_vals, sorted_indices = torch.sort(
            similarity_matrix.view(-1), descending=True
        )
        ordered_indices = -np.ones(self.n)
        rec_used = np.zeros(self.n)
        for index in sorted_indices.cpu().detach().numpy():
            source_index = int(index / self.n)
            rec_index = int(index % self.n)
            if ordered_indices[source_index] == -1 and rec_used[rec_index] == 0:
                ordered_indices[source_index] = rec_index
                rec_used[rec_index] = 1

        S_hat_reordered = S_hat[ordered_indices]
        return S_hat_reordered

    def fix_sign(self, S_hat):
        # Flatten
        S_flat = self.S.view([self.n, -1])
        S_hat_flat = S_hat.view([self.n, -1])

        if self.fix_sign_method is None:
            return S_hat
        elif self.fix_sign_method == "best":
            mse = ((S_flat - S_hat_flat) ** 2).mean(dim=-1)
            mse_inv = ((S_flat - (1 - S_hat_flat)) ** 2).mean(dim=-1)
            flip_mask = (mse_inv < mse).float().view([self.n, 1, 1, 1])
            S_hat_sign_fixed = S_hat * (1 - flip_mask) + (1 - S_hat) * flip_mask
            return S_hat_sign_fixed
        else:
            raise ValueError("Unknown method: ", self.fix_sign_method)


class ImageEval(Eval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.S = normalize_ds(self.S, method="ds", ds=self.ds)
        self.compute_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(
            self.device
        )

    def __call__(self, S_hat):
        S_hat = self.fix_order(S_hat)
        S_hat = self.fix_sign(S_hat)
        S_hat = S_hat + self.eps

        psnr_batch, ssim_batch, lpips_batch = [], [], []

        for S_hat_i, S_i in zip(S_hat.unsqueeze(1), self.S.unsqueeze(1)):
            psnr_batch.append(
                peak_signal_noise_ratio(S_hat_i.view(-1), S_i.view(-1)).cpu().item()
            )
            ssim_batch.append(
                structural_similarity_index_measure(S_hat_i, S_i).cpu().item()
            )
            lpips_batch.append(
                self.compute_lpips(normalize_lpips(S_hat_i), normalize_lpips(S_i))
                .cpu()
                .item()
            )

        metrics_batch = {
            "psnr": psnr_batch,
            "ssim": ssim_batch,
            "lpips": lpips_batch,
        }
        metrics_avg = {
            "psnr": np.mean(psnr_batch),
            "ssim": np.mean(ssim_batch),
            "lpips": np.mean(lpips_batch),
        }

        # return metrics_avg, metrics_batch, S_hat
        return S_hat
