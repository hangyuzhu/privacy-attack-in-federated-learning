"""GGL: Auditing Privacy Defenses in Federated Learning via Generative Gradient Leakage

https://arxiv.org/pdf/2203.15696

"""

import time
from tqdm import tqdm
import torch
import torch.nn as nn
import nevergrad as ng


def ggl(model, generator, gt_grads, dummy, rec_epochs, device):
    """GGL implementation

    GGL indirectly optimizes the dummy data generated from the latent features through a Generator
    Note that, the quality of Generator significantly determines the quality of the fake data
    CMA-ES is employed here as an optimizer for gradient-free stochastic search
    Loss function is constructed by DLG loss + KLD loss

    :param model: inferred model
    :param generator: trained GGL generator
    :param gt_grads: gradients of the ground truth data
    :param dummy: TorchDummy object
    :param rec_epochs: number of reconstruction epochs
    :param device: cpu or cuda
    :return: dummy data & dummy label
    """
    # be careful about the influence on model.train() & model.eval()
    # especially dropout layer and bn layer are included in model
    # we follow the original implementation to set eval() here
    model.eval()
    generator.eval()

    reconstructor = CMAReconstructor(model, generator, rec_epochs, device=device)
    dummy_data, dummy_label = reconstructor.reconstruct(gt_grads)

    dummy.append(dummy_data)
    dummy.append_label(dummy_label)
    return dummy_data, dummy_label


class CMAReconstructor:
    """ CMA-ES constructor of GGL

    The official implementation performs 500 epochs and each contains 50 sample trials
    However, we find that directly running 500x50 times would give a more stable outcome

    On the ImageNet dataset, for algorithms that do not innately support bound constraints, we
    apply the tanh function to achieve the bound.

    """

    def __init__(self, model, generator, rec_epochs=25000, search_dim=128, use_tanh=False, device="cpu"):
        self.model = model
        self.generator = generator

        parametrization = ng.p.Array(init=torch.zeros(search_dim))
        self.ng_optimizer = ng.optimizers.registry["CMA"](parametrization=parametrization, budget=rec_epochs)

        self.use_tanh = use_tanh
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100).to(device)

    def reconstruct(self, gt_grads):
        # infer label by the method introduced in iDLG
        inferred_label = torch.argmin(torch.sum(gt_grads[-2], dim=-1), dim=-1).detach().reshape((1,))
        print('Inferred label: {}'.format(inferred_label.item()))

        pbar = tqdm(range(self.ng_optimizer.budget),
                    total=self.ng_optimizer.budget,
                    desc=f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}')
        for _ in pbar:
            z = self.ng_optimizer.ask()
            loss = self.ng_loss(z=z.value, gt_grads=gt_grads, label=inferred_label)
            self.ng_optimizer.tell(z, loss)
            pbar.set_description("Loss {:.6}".format(loss))

        # get the best point
        recommendation = self.ng_optimizer.provide_recommendation()
        z_res = torch.from_numpy(recommendation.value).unsqueeze(0).to(self.device)

        if self.use_tanh:
            z_res = z_res.tanh()

        with torch.no_grad():
            # float64 -> float32
            dummy_data = self.generator(z_res.float())

        return dummy_data, inferred_label

    def ng_loss(self, z, gt_grads, label):
        """DLG loss + KLD loss

        EMA-CS is a gradient-free optimization algorithm
        Caution: 1) latent input should be converted to tensor
                 2) the loss value should be converted to float type

        :param z: latent input (ndarray)
        :param gt_grads: gradients of the ground truth data
        :param label: inferred label
        :return: loss value
        """
        z = torch.from_numpy(z).unsqueeze(0).to(self.device)
        if self.use_tanh:
            z = z.tanh()

        with torch.no_grad():
            # float64 -> float32
            x = self.generator(z.float())

        # compute the dummy gradient
        self.model.zero_grad()
        dummy_loss = self.criterion(self.model(x), label)
        dummy_grads = torch.autograd.grad(dummy_loss, self.model.parameters())
        # ema-cs is a gradient free optimization method
        dummy_grads = [grad.detach() for grad in dummy_grads]

        # calculate l2 norm
        loss = 0
        for i in range(len(dummy_grads)):
            loss += ((dummy_grads[i] - gt_grads[i]).pow(2)).sum()
        loss /= len(dummy_grads)

        if not self.use_tanh:
            KLD = -0.5 * torch.sum(
                1 + torch.log(torch.std(z.squeeze(), correction=0).pow(2) + 1e-10) - torch.mean(z.squeeze()).pow(
                    2) - torch.std(z.squeeze(), correction=0).pow(2))
            loss += 0.1 * KLD

        return loss.item()
