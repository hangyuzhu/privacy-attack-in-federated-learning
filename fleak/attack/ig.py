""" Inverting Gradients

 https://proceedings.neurips.cc/paper/2020/file/c4ede56bbd98819ae6112b20ac6bf145-Paper.pdf

 """

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from .dlg import dummy_criterion
from ..model import MetaModel


def ig_single(model, gt_grads, dummy, rec_epochs=4000, rec_lr=0.1, alpha=1e-6, device="cpu"):
    """Reconstruct an image from single gradients

    Similar to DLG based methods but adopting cosine similarity as the loss function

    :param model: inferred model
    :param gt_grads: gradients of the ground-truth data
    :param dummy: TorchDummy object
    :param rec_epochs: number of reconstruct epochs
    :param rec_lr: reconstruct learning rate
    :param alpha: hyperparameter for TV term
    :param device: cpu or cuda
    :return: dummy data & dummy label
    """
    model.eval()

    reconstructor = GradientReconstructor(model, dummy, rec_epochs, rec_lr, alpha, device)
    dummy_data, dummy_label = reconstructor.reconstruct(gt_grads)

    dummy.append(dummy_data)
    dummy.append_label(dummy_label)

    return dummy_data, dummy_label


def ig_multi(model, gt_grads, dummy, rec_epochs=8000, rec_lr=0.1,
             local_epochs=5, local_lr=1e-4, alpha=1e-6, device="cpu"):
    """Reconstruct one or multiple images from weights after several SGD steps

    Dummy gradients are simulated by multiple steps of SGD

    :param model: inferred model
    :param gt_grads: gradients of the ground-truth data
    :param dummy: TorchDummy object
    :param rec_epochs: number of reconstruct epochs
    :param rec_lr: reconstruct learning rate
    :param local_epochs: number of epochs for client training
    :param local_lr: learning rate for client training
    :param alpha: hyperparameter for TV term
    :param device: cpu or cuda
    :return: dummy data & dummy label
    """
    model.eval()

    reconstructor = FedAvgReconstructor(model, dummy, rec_epochs, rec_lr, local_epochs, local_lr, alpha, device)
    dummy_data, dummy_label = reconstructor.reconstruct(gt_grads)

    dummy.append(dummy_data)
    dummy.append_label(dummy_label)
    return dummy_data, dummy_label


class GradientReconstructor:
    """Reconstruct an image from gradient after single step of gradient descent"""

    def __init__(self, model, dummy, epochs, lr, alpha, device):
        """

        :param model: inferred model
        :param dummy: TorchDummy object
        :param epochs: reconstruct epochs
        :param lr: reconstruct learning rate
        :param alpha: hyperparameter for TV term
        :param device: cpu or cuda
        """
        self.model = model
        self.dummy = dummy
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.device = device

        # if converting labels to integers
        self.convert_label = False

    def reconstruct(self, gt_grads):
        # generate dummy data with Gaussian distribution
        dummy_data = self.dummy.generate_dummy_input(self.device)

        # server has no access to inferred data labels
        if self.dummy.batch_size == 1:
            # label prediction by iDLG
            # dummy label is not updated
            dummy_label = torch.argmin(torch.sum(gt_grads[-2], dim=-1), dim=-1).detach().reshape((1,))
            optimizer = optim.Adam([dummy_data], lr=self.lr)
            criterion = nn.CrossEntropyLoss().to(self.device)
        else:
            self.convert_label = True
            # DLG label recovery
            # dummy labels should be simultaneously updated
            dummy_label = self.dummy.generate_dummy_label(self.device)
            optimizer = optim.Adam([dummy_data, dummy_label], lr=self.lr)
            criterion = dummy_criterion

        # set learning rate decay
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[self.epochs // 2.667, self.epochs // 1.6, self.epochs // 1.142],
            gamma=0.1
        )  # 3/8 5/8 7/8

        for i in range(self.epochs):
            closure = self._gradient_closure(optimizer, criterion, gt_grads, dummy_data, dummy_label)
            rec_loss = optimizer.step(closure)
            scheduler.step()

            with torch.no_grad():
                # small trick 2: project into image space
                dummy_data.data = torch.max(
                    torch.min(dummy_data, (1 - self.dummy.t_dm) / self.dummy.t_ds), -self.dummy.t_dm / self.dummy.t_ds)
                if i + 1 == self.epochs:
                    print(f'Epoch: {i + 1}. Rec. loss: {rec_loss.item():2.4f}.')

        if self.convert_label:
            return dummy_data, torch.argmax(dummy_label, dim=-1)

        return dummy_data, dummy_label

    def _gradient_closure(self, optimizer, criterion, gt_grads, dummy_data, dummy_label):
        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()

            dummy_pred = self.model(dummy_data)
            dummy_loss = criterion(dummy_pred, dummy_label)
            dummy_grads = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)

            rec_loss = cosine_similarity_loss(dummy_grads, gt_grads)
            rec_loss += self.alpha * total_variation(dummy_data)
            rec_loss.backward()

            # small trick 1: convert the grad to 1 or -1
            dummy_data.grad.sign_()
            return rec_loss
        return closure


class FedAvgReconstructor(GradientReconstructor):
    """Reconstruct one or multiple imagex from weights after n SGD steps

    Caution: epochs & lr are hyperparameters for reconstruction updates
             while local_epochs & local_lr are hyperparameters for recovering gradients

    """

    def __init__(self, model, dummy, epochs, lr, local_epochs, local_lr, alpha, device):
        super(FedAvgReconstructor, self).__init__(
            model=model,
            dummy=dummy,
            epochs=epochs,
            lr=lr,
            alpha=alpha,
            device=device,
        )
        self.local_epochs = local_epochs
        self.local_lr = local_lr

    def _gradient_closure(self, optimizer, criterion, gt_grads, dummy_data, dummy_label):
        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()

            dummy_grads = multi_step_gradients(
                self.model, dummy_data, dummy_label, criterion, self.local_epochs, self.local_lr
            )
            rec_loss = cosine_similarity_loss(dummy_grads, gt_grads)
            rec_loss += self.alpha * total_variation(dummy_data)
            rec_loss.backward()

            # small trick 1: convert the grad to 1 or -1
            dummy_data.grad.sign_()
            return rec_loss
        return closure


def multi_step_gradients(model, inputs, labels, criterion, local_epochs, local_lr):
    """Take a few gradient descent steps to fit the model to the given input

    This method is only valid for recovering gradients computed by SGD
    Simulate the model parameters updated by several training epochs
    Caution: transfer of grad_fn is the initial consideration for this method

    :param model: inferred model
    :param inputs: input features
    :param labels: labels
    :param criterion: loss function
    :param local_epochs: client training epochs
    :param local_lr: client learning rate
    :return: list of gradient tensors
    """

    meta_model = MetaModel(model)
    # slightly faster than using OrderedDict to copy named parameters
    # but consume more device memories
    meta_model_origin = copy.deepcopy(meta_model)

    # equivalent to local client training epochs in FL
    for i in range(local_epochs):
        # using meta parameters to do forward pass
        preds = meta_model(inputs, meta_model.parameters)
        loss = criterion(preds, labels)
        # gradients are calculated upon meta parameters
        grads = torch.autograd.grad(loss, meta_model.parameters.values(), create_graph=True)

        # this method can effectively transfer the grad_fn
        meta_model.parameters = OrderedDict(
            (n, p - local_lr * g)
            for (n, p), g in zip(meta_model.parameters.items(), grads)
        )

    meta_model.parameters = OrderedDict(
         (n, p_o - p)
         for (n, p), p_o
         in zip(meta_model.parameters.items(), meta_model_origin.parameters.values())
    )
    return list(meta_model.parameters.values())


def cosine_similarity_loss(dummy_grads, grads):
    """ Compute cosine similarity value

    Compared to L2-norm loss, it can additionally capture the direction information

    :param dummy_grads: gradients of dummy data
    :param grads: gradients of the ground truth data
    :return: the loss value
    """
    # numerator
    nu = 0
    # denominator
    dn0 = 0
    dn1 = 0
    for dg, g in zip(dummy_grads, grads):
        # equivalent to the inner product of two vectors
        nu += (dg * g).sum()
        dn0 += dg.pow(2).sum()
        dn1 += g.pow(2).sum()
    loss = 1 - nu / dn0.sqrt() / dn1.sqrt()  # l2-norm
    return loss


def total_variation(x):
    """ Total variation

    https://cbio.mines-paristech.fr/~jvert/svn/bibli/local/Rudin1992Nonlinear.pdf

     """
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy
