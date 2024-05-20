import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .ig import GradientReconstructor
from .ig import total_variation


def cpa(model, gt_grads, dummy, rec_epochs, rec_lr, ne, decor, T, tv, nv, l1, device):
    model.eval()

    attacker = CocktailPartyAttack(model, dummy, rec_epochs, rec_lr, ne, decor, T, tv, nv, l1, device)
    dummy_data = attacker.reconstruct(gt_grads)

    dummy.append(dummy_data)
    return dummy_data


class CocktailPartyAttack(GradientReconstructor):

    def __init__(self, model, dummy, epochs, lr, ne, decor, T, tv, nv, l1, device):
        super(CocktailPartyAttack, self).__init__(
            model=model,
            dummy=dummy,
            epochs=epochs,
            lr=lr,
            tv=tv,
            device=device
        )
        self.ne = ne
        self.decor = decor
        self.T = T
        self.nv = nv
        self.l1 = l1
        self.a = 1
        self.eps = torch.tensor(1e-20, device=self.device)

    def reconstruct(self, gt_grads):
        # attack weights of the linear layer
        invert_grads = gt_grads[self.model.attack_index]
        # center & whiten the ground truth gradients
        grads_zc, grads_mu = self.zero_center(invert_grads)
        grads_w, w = self.whiten(grads_zc)

        U = torch.empty(
            [self.dummy.batch_size, self.dummy.batch_size],
            dtype=torch.float,
            requires_grad=True,
            device=self.device
        )
        nn.init.eye_(U)
        optimizer = optim.Adam([U], lr=self.lr, weight_decay=0)

        pbar = tqdm(range(self.epochs),
                    total=self.epochs,
                    desc=f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}')
        # optimizing unmixing matrix U
        for _ in pbar:
            optimizer.zero_grad()
            loss = self._build_loss(U, grads_w, w, grads_mu)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            U_norm = U / (U.norm(dim=-1, keepdim=True) + self.eps)
            X_hat = torch.matmul(U_norm, grads_w)
            X_hat = X_hat + torch.matmul(torch.matmul(U_norm, w), grads_mu)
            X_hat = X_hat.detach().view(self.dummy.input_shape)

        # normalize the reconstructed data
        return normalize(X_hat)

    def _build_loss(self, U, grads_w, w, grads_mu):
        """Construct the loss function of CPA

        :param U: unmixing matrix U
        :param grads_w: whitened gradients
        :param w: whiten transformation matrix
        :param grads_mu: mean of gradient outputs
        :return: loss
        """
        loss_ne = loss_decor = loss_nv = loss_tv = loss_l1 = torch.tensor(
            0.0, device=self.device
        )

        U_norm = U / (U.norm(dim=-1, keepdim=True) + self.eps)

        # Neg Entropy Loss
        X_hat = torch.matmul(U_norm, grads_w)

        if torch.isnan(X_hat).any():
            raise ValueError(f"S_hat has NaN")

        if self.ne > 0:
            # A high value of negentropy indicates a high degree of non-Gaussianity.
            loss_ne = -(((1 / self.a) * torch.log(torch.cosh(self.a * X_hat) + self.eps).mean(dim=-1)) ** 2).mean()

        # Undo centering, whitening
        X_hat = X_hat + torch.matmul(torch.matmul(U_norm, w), grads_mu)

        # Decorrelation Loss (decorrelate i-th row with j-th row, s.t. j>i)
        if self.decor > 0:
            # We assume that the source signals are independently chosen and thus their values are uncorrelated
            cos_matrix = torch.matmul(U_norm, U_norm.T).abs()
            loss_decor = (torch.exp(cos_matrix * self.T) - 1).mean()

        # Prior Loss
        if self.tv > 0 and self.nv == 0:  # if nv > 0, tv is meant for the generator
            loss_tv = total_variation(X_hat.view(self.dummy.input_shape))

        if self.nv > 0:
            # sign regularization function for leaking private embeddings
            # Minimizing loss_nv ensures that z is either non-negative or non-positive.
            loss_nv = torch.minimum(
                F.relu(-X_hat).norm(dim=-1), F.relu(X_hat).norm(dim=-1)
            ).mean()

        if self.l1 > 0:
            # l1-norm: embedding is sparse
            loss_l1 = torch.abs(X_hat).mean()

        loss = (
                loss_ne
                + (self.decor * loss_decor)
                + (self.tv * loss_tv)
                + (self.nv * loss_nv)
                + (self.l1 * loss_l1)
        )
        return loss

    @staticmethod
    def zero_center(x):
        # centering across the input dimension of the weights
        x_mu = x.mean(dim=-1, keepdims=True)  # channel first
        return x - x_mu, x_mu

    def whiten(self, x):
        """Whitening the gradients

        Purpose: 1) Project the dataset onto the eigenvectors.
                    This rotates the dataset so that there is no correlation between the components.
                 2) Normalize the dataset to have a variance of 1 for all components.

        Due to the 'channel first' characteristics of cudnn,
        transpose operation should be applied to PCA processing

        :param x: centered gradients
        :return: whitened gradients & normalized eigenvectors
        """
        cov = torch.matmul(x, x.T) / (x.shape[1] - 1)
        eig_vals, eig_vecs = torch.linalg.eig(cov)
        # select top k (k = batch_size) of eigenvectors
        # make sure the output dimension of the linear layer is larger than batch size
        topk_indices = torch.topk(eig_vals.float().abs(), self.dummy.batch_size)[1]

        lamb = eig_vals.float()[topk_indices].abs()
        # whiten transformation: normalize the dataset to have a variance of 1 for all components.
        lamb_inv_sqrt = torch.diag(1 / (torch.sqrt(lamb) + self.eps)).float()
        W = torch.matmul(lamb_inv_sqrt, eig_vecs.float().T[topk_indices]).float()
        x_w = torch.matmul(W, x)
        return x_w, W


def normalize(inp, method=None):
    if method is None:
        pass
    elif method == "infer":
        orig_shape = inp.shape
        n = orig_shape[0]
        inp = inp.view([n, -1])
        inp = (inp - inp.min(dim=-1, keepdim=True)[0]) / (
            inp.max(dim=-1, keepdim=True)[0] - inp.min(dim=-1, keepdim=True)[0]
        )
        inp = inp.view(orig_shape)
    else:
        raise ValueError(f"Unknown method {method}")
    return inp
