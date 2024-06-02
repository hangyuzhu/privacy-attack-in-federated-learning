"""Cocktail Party Attack

https://proceedings.mlr.press/v202/kariyappa23a/kariyappa23a.pdf

"""

import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .ig import GradientReconstructor
from .ig import total_variation


def cpa(model, gt_grads, dummy, rec_epochs, rec_lr, fi_lr, decor, T, tv, nv, l1, fi, device):
    """Cocktail Party Attack (CPA)

    The core idea is to adopt independent component analysis (ICA) to recover inputs from aggregated gradients
    1) For MLPs, utilize CPA to directly recover private inputs
    2) For CNNs, utilize CPA to recover embeddings, and further adopt feature inversion to recover inputs

    :param model: inferred model
    :param gt_grads: gradients of ground-truth data
    :param dummy: TorchDummy object
    :param rec_epochs: reconstruct epochs
    :param rec_lr: reconstruct learning rate
    :param fi_lr: learning rate of feature inversion
    :param decor: decorrelation weight
    :param T: temperature for cosine similarity when computing decor loss in CPA
    :param tv: total Variation prior weight
    :param nv: negative value penalty
    :param l1: hyperparameter of l1-norm
    :param fi: feature inversion weight
    :param device: cpu or cuda
    :return: dummy data
    """
    # ensure the model type
    assert model.model_type in ["cpa_cov", "cpa_fc2"]
    model.eval()

    if model.model_type == "cpa_fc2":
        inp_type = "image"
    elif model.model_type == "cpa_cov":
        inp_type = "emb"
    else:
        raise ValueError(f"Unexpected model type {model.model_type}")

    gi = CocktailPartyAttack(model, dummy, rec_epochs, inp_type, rec_lr, decor, T, tv, nv, l1, device)
    rec_gi = gi.reconstruct(gt_grads)

    if inp_type == "emb":
        fi = FeatureInversionAttack(model, dummy, rec_epochs, fi_lr, fi, tv, device)
        # make recovered embeds positive
        rec_fi = fi.reconstruct(rec_gi.abs())
        dummy_data = rec_fi
    else:
        dummy_data = rec_gi

    dummy.append(dummy_data, method="infer")
    return dummy_data


class CocktailPartyAttack(GradientReconstructor):
    """

    Frame a BSS problem and adapt ICA to recover the private inputs
    from aggregate gradient/weight updates

    """

    def __init__(self, model, dummy, epochs, inp_type, lr, decor, T, tv, nv, l1, device):
        """

        :param model: inferred model
        :param dummy: TorchDummy object
        :param epochs: reconstruct epochs
        :param inp_type: image & emb
        :param lr: reconstruct learning rate
        :param decor: decorrelation weight (CPA)
        :param T: temperature for cosine similarity when computing decor loss in CPA
        :param tv: total Variation prior weight
        :param nv: negative value penalty
        :param l1: hyperparameter of l1-norm
        :param device: cpu or cuda
        """
        assert inp_type in ["image", "emb"]
        super(CocktailPartyAttack, self).__init__(
            model=model,
            dummy=dummy,
            epochs=epochs,
            lr=lr,
            tv=tv,
            device=device
        )
        self.inp_type = inp_type
        if self.inp_type == "image":
            self.inp_shape = dummy.input_shape
        else:
            self.inp_shape = [dummy.batch_size, -1]
        self.decor = decor
        self.T = T
        self.nv = nv
        self.l1 = l1
        self.a = 1
        self.eps = torch.tensor(1e-20, device=self.device)

    def reconstruct(self, gt_grads):
        """Reconstruct private inputs or embeddings

        :param gt_grads: gradients of ground truth data
        :return: Private inputs or embeddings
        """
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
            # undo whitening & centering
            X_hat = X_hat + torch.matmul(torch.matmul(U_norm, w), grads_mu)
            X_hat = X_hat.detach().view(self.inp_shape)
            if self.inp_type == "image":
                # normalize the reconstructed image data
                X_hat = normalize(X_hat, self.dummy, method="infer")

        return X_hat

    def _build_loss(self, U, grads_w, w, grads_mu):
        """Construct the loss function of CPA

        1) Reconstruct image: NE loss + MI loss + TV loss
        2) Reconstruct embed: NE loss + MI loss + NV loss + l1 norm

        :param U: unmixing matrix U
        :param grads_w: whitened gradients
        :param w: normalized eigenvectors
        :param grads_mu: mean of gradient outputs
        :return: loss
        """
        loss_decor = loss_nv = loss_tv = loss_l1 = torch.tensor(
            0.0, device=self.device
        )

        # small trick
        U_norm = U / (U.norm(dim=-1, keepdim=True) + self.eps)

        # Neg Entropy Loss
        X_hat = torch.matmul(U_norm, grads_w)

        if torch.isnan(X_hat).any():
            raise ValueError(f"S_hat has NaN")

        # A high value of negentropy indicates a high degree of non-Gaussianity.
        loss_ne = -(((1 / self.a) * torch.log(torch.cosh(self.a * X_hat) + self.eps).mean(dim=-1)) ** 2).mean()

        # Undo centering, whitening
        X_hat = X_hat + torch.matmul(torch.matmul(U_norm, w), grads_mu)

        # Decorrelation Loss (decorrelate i-th row with j-th row, s.t. j>i)
        # Mutual Independence (MI): We assume that the source
        # signals are independently chosen and thus their values are uncorrelated
        if self.decor > 0:
            # We assume that the source signals are independently chosen and thus their values are uncorrelated
            cos_matrix = torch.matmul(U_norm, U_norm.T).abs()
            loss_decor = (torch.exp(cos_matrix * self.T) - 1).mean()

        # Prior Loss
        if self.tv > 0 and self.nv == 0:  # if nv > 0, tv is meant for the generator
            loss_tv = total_variation(X_hat.view(self.inp_shape))

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
        Caution: Computed eigenvectors are complex numbers

        Due to the 'channel first' characteristics of cudnn,
        transpose operation should be applied to PCA processing

        :param x: centered gradients with shape (d_out, d_in)
        :return: whitened gradients & normalized eigenvectors
        """
        cov = torch.matmul(x, x.T) / (x.shape[1] - 1)
        eig_vals, eig_vecs = torch.linalg.eig(cov)
        # select top k (k = batch_size) of eigenvectors
        # make sure the output dimension of the linear layer is larger than batch size
        topk_indices = torch.topk(eig_vals.float().abs(), self.dummy.batch_size)[1]

        lamb = eig_vals.float()[topk_indices].abs()
        # whiten transformation: normalize the dataset to have a variance of 1 for all components.
        lamb_inv_sqrt = torch.diag(1 / (torch.sqrt(lamb) + self.eps)).float()  # b x b
        n_eig_vecs = torch.matmul(lamb_inv_sqrt, eig_vecs.float().T[topk_indices]).float()  # b x b * b x d_out
        x_w = torch.matmul(n_eig_vecs, x)
        return x_w, n_eig_vecs


class FeatureInversionAttack:
    """ Invert the embedding produced by a neural network to recover the input. """

    def __init__(self, model, dummy, rec_epochs, fi_lr, fi, tv, device):
        """

        :param model: inferred model
        :param dummy: TorchDummy object
        :param rec_epochs: reconstruct epochs
        :param fi_lr: learning rate of feature inversion
        :param fi: feature inversion weight
        :param tv: total variation prior weight
        :param device: cpu or cuda
        """
        self.model = model
        self.dummy = dummy
        self.rec_epochs = rec_epochs
        self.fi_lr = fi_lr
        self.fi = fi
        self.tv = tv

        self.device = device

        for p in self.model.parameters():
            p.requires_grad = False

    def reconstruct(self, rec_z):
        dummy_data = self.dummy.generate_dummy_input(device=self.device)
        optimizer = optim.Adam([dummy_data], lr=self.fi_lr, weight_decay=0)
        cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-10).to(self.device)

        # optimizer the fake data through reconstructed latent inputs
        pbar = tqdm(range(self.rec_epochs),
                    total=self.rec_epochs,
                    desc=f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}')
        for _ in pbar:
            optimizer.zero_grad()
            loss = self._build_loss(cosine_similarity, rec_z, dummy_data)
            loss.backward()
            optimizer.step()

        # small trick
        dummy_data = normalize(dummy_data.detach(), self.dummy, method="ds")
        return dummy_data

    def _build_loss(self, cs_criterion, rec_z, dummy_data):
        loss_fi = loss_tv = torch.tensor(0.0, device=self.device)

        # Box Image (small trick)
        if self.dummy.normalize:
            dummy_data.data = torch.max(
                torch.min(dummy_data, (1 - self.dummy.t_dm) / self.dummy.t_ds),
                -self.dummy.t_dm / self.dummy.t_ds
            )
        else:
            dummy_data.data = torch.clamp(dummy_data, 0, 1)

        _, z_hat = self.model(dummy_data, return_z=True)
        if self.fi > 0:
            loss_fi = (1 - cs_criterion(rec_z, z_hat)).mean()

        if self.tv > 0:
            loss_tv = total_variation(dummy_data)

        loss = self.fi * loss_fi + self.tv * loss_tv
        return loss


def normalize(inp, dummy, method=None):
    """Normalize data

    infer: trick proposed by CPA
    ds: trick proposed by inverting gradients

    :param inp: input data
    :param dummy: TorchDummy object
    :param method: infer or ds
    :return: normalized data
    """
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
    elif method == "ds":
        if dummy.normalize:
            inp = torch.clamp((inp * dummy.t_ds) + dummy.t_dm, 0, 1)
        else:
            inp = torch.clamp(inp, 0, 1)
    else:
        raise ValueError(f"Unknown method {method}")
    return inp
