import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .ig import GradientReconstructor
from .ig import total_variation


def cpa(model, gt_grads, dummy, device):
    model.eval()


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

        # exp_path = get_attack_exp_path(self.args)
        # self.w_pkl_file = f"{exp_path}/w_{self.n_comp}_{self.batch_id}.pkl"
        # self.X = self.grads[self.model.attack_index]
        # self.X_zc, self.X_mu = self.zero_center(self.X)
        #
        # if exists(self.w_pkl_file):
        #     w_data = read_pickle(self.w_pkl_file)
        #     self.X_w, self.W_w = w_data["X_w"].to(self.device), w_data["W_w"].to(
        #         self.device
        #     )
        #     print(f"loaded w data from {self.w_pkl_file}")
        # else:
        #     self.X_w, self.W_w = self.whiten(self.X_zc)
        #     write_pickle(
        #         {"X_w": self.X_w.detach().cpu(), "W_w": self.W_w.detach().cpu()},
        #         self.w_pkl_file,
        #     )
        #
        # self.W_hat = torch.empty(
        #     [self.n_comp, self.n_comp],
        #     dtype=torch.float,
        #     requires_grad=True,
        #     device=self.device,
        # )
        # torch.nn.init.eye_(self.W_hat)
        # param_list = [self.W_hat]
        #
        # self.opt = get_opt(param_list, self.args.opt, lr=self.args.lr)
        # self.sch = get_sch(self.args.sch, self.opt, epochs=self.args.n_iter)
        # self.a = 1

    # def set_inp_type(self):
    #     if ds_type_dict[self.ds] == "image" and self.model.model_type == "fc":
    #         self.inp_type = "image"
    #         self.inp_shape = [self.n_comp] + xshape_dict[self.ds]
    #     else:
    #         self.inp_type = "emb"
    #         self.inp_shape = [self.n_comp, -1]

    def zero_center(self, x):
        x_mu = x.mean(dim=-1, keepdims=True)
        return x - x_mu, x_mu

    def whiten(self, x):
        cov = torch.matmul(x, x.T) / (x.shape[1] - 1)
        eig_vals, eig_vecs = torch.linalg.eig(cov)
        topk_indices = torch.topk(eig_vals.float().abs(), self.dummy.batch_size)[1]

        lamb = eig_vals.float()[topk_indices].abs()
        lamb_inv_sqrt = torch.diag(1 / (torch.sqrt(lamb) + self.eps)).float()
        W = torch.matmul(lamb_inv_sqrt, eig_vecs.float().T[topk_indices]).float()
        x_w = torch.matmul(W, x)
        return x_w, W

    def reconstruct(self, gt_grads):
        # for batch in range(attack_log.batch, args.n_batch):
        #     # get a batch of inputs
        #     inp_key = "x"
        #     inp = torch.tensor(grad_data[inp_key][batch], device=device)
        #     emb = torch.tensor(
        #         grad_data["z"][batch] if len(grad_data["z"]) > 0 else [], device=device
        #     )
        #     labels = torch.tensor(grad_data["y"][batch], device=device)
        #     grads = [torch.tensor(g, device=device) for g in grad_data["grad"][batch]]
        #
        #     attack_log.update_batch(batch, inp, emb, grads)
        #
        #     if attack_log.attack_mode == "gi":
        #         # === Gradient Inversion ===
        #         gi = get_gi(args.attack, model, grads, labels, args, batch, attack_log)
        #         pbar = get_pbar(range(gi.start_iter, args.n_iter), disable=True)
        #         eval = get_eval(
        #             inp, emb, model.model_type, ds_type_dict[args.ds], args.attack, fi=False
        #         )

        invert_grads = gt_grads[self.model.attack_index]
        g_zc, g_mu = self.zero_center(invert_grads)
        g_w, w_w = self.whiten(g_zc)

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
        for iter in pbar:
            loss = self._gradient_closure(optimizer, U, g_w, w_w, g_mu)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            U_norm = U / (U.norm(dim=-1, keepdim=True) + self.eps)
            X_hat = torch.matmul(U_norm, g_w)
            X_hat = X_hat + torch.matmul(torch.matmul(U_norm, w_w), g_mu)
            X_hat = X_hat.detach().view(self.dummy.input_shape)
            X_hat = normalize(X_hat)

        return X_hat

    def _gradient_closure(self, optimizer, U, g_w, w_w, g_mu, *args):
        loss_ne = loss_decor = loss_nv = loss_tv = loss_l1 = torch.tensor(
            0.0, device=self.device
        )

        optimizer.zero_grad()
        U_norm = U / (U.norm(dim=-1, keepdim=True) + self.eps)

        # Neg Entropy Loss
        X_hat = torch.matmul(U_norm, g_w)

        if torch.isnan(X_hat).any():
            raise ValueError(f"S_hat has NaN")

        if self.ne > 0:
            loss_ne = -(((1 / self.a) * torch.log(torch.cosh(self.a * X_hat) + self.eps).mean(dim=-1)) ** 2).mean()

        # Undo centering, whitening
        X_hat = X_hat + torch.matmul(torch.matmul(U_norm, w_w), g_mu)

        # Decorrelation Loss (decorrelate i-th row with j-th row, s.t. j>i)
        if self.decor > 0:
            cos_matrix = torch.matmul(U_norm, U_norm.T).abs()
            loss_decor = (torch.exp(cos_matrix * self.T) - 1).mean()

        # Prior Loss
        if self.tv > 0 and self.nv == 0:  # if nv > 0, tv is meant for the generator
            loss_tv = total_variation(X_hat.view(self.dummy.input_shape))

        if self.nv > 0:
            loss_nv = torch.minimum(
                F.relu(-X_hat).norm(dim=-1), F.relu(X_hat).norm(dim=-1)
            ).mean()

        if self.l1 > 0:
            loss_l1 = torch.abs(X_hat).mean()

        loss = (
                loss_ne
                + (self.decor * loss_decor)
                + (self.tv * loss_tv)
                + (self.nv * loss_nv)
                + (self.l1 * loss_l1)
        )
        return loss

        # loss.backward()
        # self.opt.step()
        # if self.sch:
        #     self.sch.step()
        # loss_dict = self.make_dict(
        #     [
        #         "loss",
        #         "loss_ne",
        #         "loss_decor",
        #         "loss_tv",
        #         "loss_nv",
        #         "loss_l1",
        #     ],
        #     [
        #         loss,
        #         loss_ne,
        #         loss_decor,
        #         loss_tv,
        #         loss_nv,
        #         loss_l1,
        #     ],
        # )
        #
        # return loss_dict


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
