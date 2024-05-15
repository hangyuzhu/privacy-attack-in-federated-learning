import time
from tqdm import tqdm
import torch
import torch.nn as nn
import nevergrad as ng


def ggl(model, generator, gt_grads, dummy, rec_epochs, device):
    model.eval()
    generator.eval()

    reconstructor = CMAReconstructor(model, generator, rec_epochs, device=device)
    dummy_data, dummy_label = reconstructor.reconstruct(gt_grads)

    dummy.append(dummy_data)
    dummy.append_label(dummy_label)
    return dummy_data, dummy_label


class CMAReconstructor:
    """
    Reconstruction for WGAN-GP

    """

    def __init__(self, model, generator, rec_epochs=25000, search_dim=128, use_tanh=False, device="cpu"):
        self.model = model
        self.generator = generator

        # self.search_dim = search_dim
        self.use_tanh = use_tanh

        parametrization = ng.p.Array(init=torch.zeros(search_dim))
        self.ng_optimizer = ng.optimizers.registry["CMA"](parametrization=parametrization, budget=rec_epochs)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100).to(device)

        self.device = device

    def reconstruct(self, gt_grads):
        labels = torch.argmin(torch.sum(gt_grads[-2], dim=-1), dim=-1).detach().reshape((1,))
        print('Inferred label: {}'.format(labels))

        pbar = tqdm(range(self.ng_optimizer.budget),
                    total=self.ng_optimizer.budget,
                    desc=f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}')
        for _ in pbar:
            z = self.ng_optimizer.ask()
            loss = self.ng_loss(z=z.value, input_gradient=gt_grads, labels=labels)
            self.ng_optimizer.tell(z, loss)
            pbar.set_description("Loss {:.6}".format(loss))

        recommendation = self.ng_optimizer.provide_recommendation()
        z_res = torch.from_numpy(recommendation.value).unsqueeze(0).to(self.device)

        if self.use_tanh:
            z_res = z_res.tanh()

        with torch.no_grad():
            x_res = self.generator(z_res.float())

        return x_res, labels

    def ng_loss(
        self,
        z,  # latent variable to be optimized
        input_gradient,
        labels
    ):
        z = torch.Tensor(z).unsqueeze(0).to(input_gradient[0].device)
        if self.use_tanh:
            z = z.tanh()

        with torch.no_grad():
            x = self.generator(z)

        # compute the trial gradient
        self.model.zero_grad()
        target_loss = self.criterion(self.model(x), labels)
        trial_gradient = torch.autograd.grad(target_loss, self.model.parameters())
        trial_gradient = [grad.detach() for grad in trial_gradient]

        # calculate l2 norm
        dist = 0
        for i in range(len(trial_gradient)):
            dist += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum()
        dist /= len(trial_gradient)

        if not self.use_tanh:
            KLD = -0.5 * torch.sum(
                1 + torch.log(torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(
                    z.squeeze(),
                    axis=-1).pow(
                    2) - torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2))

            dist += 0.1 * KLD

        return dist.item()
