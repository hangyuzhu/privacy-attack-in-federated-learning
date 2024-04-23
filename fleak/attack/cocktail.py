import torch
from abc import abstractmethod
from torch.optim import Adam, SGD
from ..data.image_dataset import xshape_dict,ds_mean,ds_std
from ..attack.utils import ImageEval
import torch.nn.functional as F
from tqdm import tqdm

default_config = dict(
    tv=3.1,
    l1=0,
    nv=0,
    ne=1,
    T=12.4,
    decor=1.47,
    use_labels=False,
    n_iter=25000,
    sigma=0.0,
    C=1.0,
    lr=0.001
)

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

## GradInversion
class GIAttack:
    # def __init__(self,  model, grads, iters, batch_size, fc_index, lr, ds, device):
    def __init__(self,  model, grads, batch_size, fc_index, config, ds, device):
        self.model = model
        self.model.train()
        self.shared_grads= grads
        self.use_labels=config["use_labels"]
        self.device = device
        self.n_iter = config["n_iter"]
        self.n_comp = batch_size
        self.fc_index = fc_index
        self.sigma = config["sigma"] * config["C"]
        self.lr = config["lr"]
        self.eps = torch.tensor(1e-20, device=self.device)
        self.ds = ds

        self.tv = config["tv"]
        self.l1 = config["l1"]
        self.nv = config["nv"]
        self.ne = config["ne"]
        self.T = config["T"]
        self.decor = config["decor"]
        self.set_inp_type()
        self.mean = torch.tensor(ds_mean[self.ds], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(ds_std[self.ds], device=self.device).view(1, 3, 1, 1)
        self.start_iter = 0


    def total_variation(self, x):
        dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return dx + dy

    def make_dict(self, keys, vals):
        return {k: v.cpu().detach().item() for k, v in zip(keys, vals)}

    @abstractmethod
    def set_inp_type(self):
        pass


class CocktailAttack(GIAttack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.X = self.shared_grads[self.fc_index]
        self.X_zc, self.X_mu = self.zero_center(self.X)

        self.X_w, self.W_w = self.whiten(self.X_zc)

        self.W_hat = torch.empty(
            [self.n_comp, self.n_comp],
            dtype=torch.float,
            requires_grad=True,
            device=self.device,
        )
        torch.nn.init.eye_(self.W_hat)
        param_list = [self.W_hat]

        self.optimizer = Adam(param_list, lr=self.lr, weight_decay=0)
        self.sch = None
        self.a = 1

    def set_inp_type(self):
        self.inp_tpye = "Image"
        self.inp_shape = [self.n_comp] + xshape_dict[self.ds]

    def zero_center(self, x):
        x_mu = x.mean(dim=-1, keepdims=True)
        return x - x_mu, x_mu

    def whiten(self, x):
        cov = torch.matmul(x, x.T) / (x.shape[1] -1)
        eig_vals, eig_vecs = torch.linalg.eig(cov)
        topk_indices = torch.topk(eig_vals.float().abs(), self.n_comp)[1]

        U = eig_vecs.float()
        lamb = eig_vals.float()[topk_indices].abs()
        lamb_inv_sqrt = torch.diag(1 / (torch.sqrt(lamb) + self.eps)).float()
        W = torch.matmul(lamb_inv_sqrt, U.T[topk_indices]).float()
        x_w = torch.matmul(W, x)
        return x_w, W

    def get_attack_state(self):
        state_dict = {}
        state_dict["W_hat"] = self.W_hat
        state_dict["opt"] = self.optimizer.state_dict()
        if self.sch is not None:
            state_dict["sch"] = self.sch.state_dict()
        return state_dict

    def set_attack_state(self, state_dict):
        self.W_hat.data = state_dict["W_hat"].data
        self.optimizer.load_state_dict(state_dict["opt"])
        if self.sch is not None:
            self.sch.load_state_dict(state_dict["sch"])

    def step(self):
        loss_ne = loss_decor = loss_nv = loss_tv = loss_l1 = torch.tensor(
            0.0, device=self.device
        )

        self.optimizer.zero_grad()
        W_hat_norm = self.W_hat / (self.W_hat.norm(dim=-1, keepdim=True) + self.eps)

        # Neg Entropy Loss
        X_w = self.X_w
        S_hat = torch.matmul(W_hat_norm, X_w)

        if torch.isnan(S_hat).any():
            raise ValueError(f"S_hat has NaN")

        if self.ne > 0:
            loss_ne = -(
                    (
                            (1 / self.a)
                            * torch.log(torch.cosh(self.a * S_hat) + self.eps).mean(dim=-1)
                    )
                    ** 2
            ).mean()
            loss_ne = torch.tensor(0.0, device=self.device)

        # Undo centering, whitening
        S_hat = S_hat + torch.matmul(torch.matmul(W_hat_norm, self.W_w), self.X_mu)

        # Decorrelation Loss (decorrelate i-th row with j-th row, s.t. j>i)
        if self.decor > 0:
            cos_matrix = torch.matmul(W_hat_norm, W_hat_norm.T).abs()
            loss_decor = (torch.exp(cos_matrix * self.T) - 1).mean()

        # Prior Loss
        if self.tv > 0 and self.nv == 0:  # if nv > 0, tv is meant for the generator
            loss_tv = self.total_variation(S_hat.view(self.inp_shape))

        if self.nv > 0:
            loss_nv = torch.minimum(
                F.relu(-S_hat).norm(dim=-1), F.relu(S_hat).norm(dim=-1)
            ).mean()

        if self.l1 > 0:
            loss_l1 = torch.abs(S_hat).mean()

        loss = (
                loss_ne
                + (self.decor * loss_decor)
                + (self.tv * loss_tv)
                + (self.nv * loss_nv)
                + (self.l1 * loss_l1)
        )

        loss.backward()
        self.optimizer.step()
        if self.sch:
            self.sch.step()
        loss_dict = self.make_dict(
            [
                "loss",
                "loss_ne",
                "loss_decor",
                "loss_tv",
                "loss_nv",
                "loss_l1",
            ],
            [
                loss,
                loss_ne,
                loss_decor,
                loss_tv,
                loss_nv,
                loss_l1,
            ],
        )

        return loss_dict

    def get_rec(self):
        with torch.no_grad():
            W_hat_norm = self.W_hat / (self.W_hat.norm(dim=-1, keepdim=True) + self.eps)
            S_hat = torch.matmul(W_hat_norm, self.X_w)
            S_hat = S_hat + torch.matmul(torch.matmul(W_hat_norm, self.W_w), self.X_mu)
            S_hat = S_hat.detach().view(self.inp_shape)
            S_hat = normalize(S_hat, method="infer")
        return S_hat


## Feature Inversion
def subsample(input_list, n, n_sample):
    if n <= n_sample:
        return input_list
    else:
        assert n_sample < n
        return [input[:n_sample] for input in input_list]

##pbar
def get_pbar(iter, disable=False, enu=False):
    t = len(iter)
    if enu:
        iter = enumerate(iter)
    if disable:
        pbar = tqdm(iter, total=t, leave=False, disable=True)
    else:
        pbar = tqdm(iter, total=t, leave=False)
    return pbar

def cocktail(model, grads, batch_size, fc_index, lr, ds="tiny_imagenet", device="CPU", if_model_conv = False):
    """
        https://proceedings.mlr.press/v202/kariyappa23a.html
        :param model: global model
        :param grads: uploaded gradients
        :param batch_size:
        :param n_sample_fi:
        :param fc_index: the index of the fc layer to attack
        :param lr: learning rate
        :param ds: datasets(type:str)
        :param device: cpu or cuda
        :return: reconstructed data
    """
    config = dict(
        tv=3.1,
        l1=0,
        nv=0,
        ne=1,
        T=12.4,
        decor=1.47,
        use_labels=False,
        n_iter=25000,
        sigma=0.0,
        C=1.0,
        lr=lr,
        n_log=5000
    )
    model.eval()
    gradients = []
    for key, value in grads.items():
        gradients.append(value)
    # gradients = torch.tensor(grads.values(), device=device)
    emb = torch.tensor([], device=device)

    gi = CocktailAttack(model, gradients, batch_size, fc_index, config, ds, device)
    pbar = get_pbar(range(gi.start_iter, config["n_iter"]), disable=True)
    # eval = ImageEval(inp, fix_order_method="ssim", fix_sign_method="best")  ##没办法获得原始数据，所以eval做不了
    for iter in pbar:
        if (iter % config["n_log"] == 0) or (iter == config["n_iter"] - 1):
            rec_gi = gi.get_rec()
    if if_model_conv:

        pass
    return rec_gi


