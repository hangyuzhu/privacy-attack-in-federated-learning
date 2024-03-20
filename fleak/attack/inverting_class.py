import torch
from copy import deepcopy
import time
from collections import defaultdict, OrderedDict

from .metrics import InceptionScore, total_variation
from .utils import label_to_onehot, MedianPool2d
from .modules import MetaMonkey


DEFAULT_CONFIG = dict(signed=False,
                      boxed=True,
                      cost_fn='sim',
                      indices='def',
                      weights='equal',
                      lr=0.1,
                      optim='adam',
                      restarts=1,
                      max_iterations=4800,
                      total_variation=1e-1,
                      init='randn',
                      filter='none',
                      lr_decay=True,
                      scoring_choice='loss')


def validate_config(config):
    for key in DEFAULT_CONFIG.keys():
        if config.get(key) is None:
            config[key] = DEFAULT_CONFIG[key]
    for key in config.keys():
        if DEFAULT_CONFIG.get(key) is None:
            raise ValueError(f'Deprecated key in config dict: {key}!')
    return config


class GradientReconstructor:
    """
    Based on IDLG algorithm, but with different loss function (Sim)
    """
    def __init__(self, model, mean_std, config, num_images=1, device="cpu"):
        self.config = validate_config(config)
        self.model = model    # global model
        self.setup = dict(device=device, dtype=next(model.parameters()).dtype)

        self.mean_std = mean_std
        self.num_images = num_images

        if self.config['scoring_choice'] == 'inception':
            self.inception = InceptionScore(batch_size=1, setup=self.setup)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean').to(device)

    def reconstruct(self, gt_grads, pred_labels, img_shape=(3, 32, 32), dryrun=False, eval=True, tol=None):
        # input data: gradients of self.model with respect to 1 gt data
        """Reconstruct image from gradient."""
        start_time = time.time()
        if eval:
            self.model.eval()

        stats = defaultdict(list)
        all_dummy_data = torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup) # 32, 1, 3,32,32
        scores = torch.zeros(self.config['restarts'])  # 32

        assert pred_labels.shape[0] == self.num_images   # 1

        try:
            for trial in range(self.config['restarts']):  # 32
                dummy_data, pred_labels = self._run_trial(all_dummy_data[trial], gt_grads, pred_labels, dryrun=dryrun)
                # Finalize
                scores[trial] = self._score_trial(dummy_data, gt_grads, pred_labels)
                all_dummy_data[trial] = dummy_data
                if tol is not None and scores[trial] <= tol:
                    break
                if dryrun:
                    break
        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass

        # Choose optimal result:
        if self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            optimal_dummy_data, stats = self._average_trials(all_dummy_data, pred_labels, gt_grads, stats)
        else:
            print('Choosing optimal result ...')
            scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
            optimal_index = torch.argmin(scores)
            print(f'Optimal result score: {scores[optimal_index]:2.4f}')
            stats['opt'] = scores[optimal_index].item()
            optimal_dummy_data = all_dummy_data[optimal_index]

        print(f'Total time: {time.time()-start_time}.')
        return optimal_dummy_data.detach(), stats

    def _run_trial(self, dummy_data: torch.Tensor, gt_grads: OrderedDict, pred_labels: torch.Tensor, dryrun=False):
        dummy_data.requires_grad = True
        if self.config['optim'] == 'adam':
            optimizer = torch.optim.Adam([dummy_data], lr=self.config['lr'])
        elif self.config['optim'] == 'sgd':
            optimizer = torch.optim.SGD([dummy_data], lr=0.01, momentum=0.9, nesterov=True)
        elif self.config['optim'] == 'LBFGS':
            optimizer = torch.optim.LBFGS([dummy_data])
        else:
            raise ValueError()

        max_iterations = self.config['max_iterations']
        dm, ds = self.mean_std
        if self.config['lr_decay']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[max_iterations // 2.667, max_iterations // 1.6,

                                                                         max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
        try:
            for iteration in range(max_iterations):  # 1000
                closure = self._gradient_closure(optimizer, dummy_data, gt_grads, pred_labels)
                rec_loss = optimizer.step(closure)
                if self.config['lr_decay']:
                    scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.config['boxed']:
                        dummy_data.data = torch.max(torch.min(dummy_data, (1 - dm) / ds), -dm / ds)

                    # if (iteration + 1 == max_iterations) or iteration % 500 == 0:
                    if iteration + 1 == max_iterations:
                        print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')

                    if (iteration + 1) % 500 == 0:
                        if self.config['filter'] == 'none':
                            pass
                        elif self.config['filter'] == 'median':
                            dummy_data.data = MedianPool2d(kernel_size=3, stride=1, padding=1, same=False)(dummy_data)
                        else:
                            raise ValueError()

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
        return dummy_data.detach(), pred_labels

    def _gradient_closure(self, optimizer, dummy_data, gt_grads, pred_labels):
        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            dummy_pred = self.model(dummy_data)
            loss = self.loss_fn(dummy_pred, pred_labels)
            dummy_grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            # cost_fn sim, indices def, weights equal
            rec_loss = reconstruction_costs([dummy_grads], gt_grads,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])

            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * total_variation(dummy_data)
            rec_loss.backward()
            if self.config['signed']:
                dummy_data.grad.sign_()
            return rec_loss
        return closure

    def _score_trial(self, dummy_data, gt_grads, pred_labels):
        if self.config['scoring_choice'] == 'loss':
            self.model.zero_grad()
            dummy_data.grad = None
            loss = self.loss_fn(self.model(dummy_data), pred_labels)
            dummy_grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            return reconstruction_costs([dummy_grads], gt_grads,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights=self.config['weights'])
        elif self.config['scoring_choice'] == 'tv':
            return total_variation(dummy_data)
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(dummy_data)
        elif self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            return 0.0
        else:
            raise ValueError()

    def _average_trials(self, x, labels, input_data, stats):
        print(f'Computing a combined result via {self.config["scoring_choice"]} ...')
        if self.config['scoring_choice'] == 'pixelmedian':
            x_optimal, _ = x.median(dim=0, keepdims=False)
        elif self.config['scoring_choice'] == 'pixelmean':
            x_optimal = x.mean(dim=0, keepdims=False)

        self.model.zero_grad()
        loss = self.loss_fn(self.model(x_optimal), labels)
        gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
        stats['opt'] = reconstruction_costs([gradient], input_data,
                                            cost_fn=self.config['cost_fn'],
                                            indices=self.config['indices'],
                                            weights=self.config['weights'])
        print(f'Optimal result score: {stats["opt"]:2.4f}')
        return x_optimal, stats


class FedAvgReconstructor(GradientReconstructor):
    """Reconstruct an image from weights after n gradient descent steps."""

    def __init__(self, model, mean_std=(0.0, 1.0), local_steps=2, local_lr=1e-4,
                 config=DEFAULT_CONFIG, num_images=1, use_updates=True, batch_size=0,device="cpu"):
        """Initialize with model, (mean, std) and config."""
        super().__init__(model, mean_std, config, num_images,device)
        self.local_steps = local_steps
        self.local_lr = local_lr
        self.use_updates = use_updates
        self.batch_size = batch_size

    def _gradient_closure(self, optimizer, x_trial, input_parameters, labels):
        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            parameters = loss_steps(self.model, x_trial, labels, loss_fn=self.loss_fn,
                                    local_steps=self.local_steps, lr=self.local_lr,
                                    use_updates=self.use_updates,
                                    batch_size=self.batch_size)
            rec_loss = reconstruction_costs([parameters], input_parameters,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])

            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * total_variation(x_trial)
            rec_loss.backward()
            if self.config['signed']:
                x_trial.grad.sign_()
            return rec_loss
        return closure

    def _score_trial(self, x_trial, input_parameters, labels):
        if self.config['scoring_choice'] == 'loss':
            self.model.zero_grad()
            parameters = loss_steps(self.model, x_trial, labels, loss_fn=self.loss_fn,
                                    local_steps=self.local_steps, lr=self.local_lr, use_updates=self.use_updates)
            return reconstruction_costs([parameters], input_parameters,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights=self.config['weights'])
        elif self.config['scoring_choice'] == 'tv':
            return total_variation(x_trial)
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(x_trial)


def loss_steps(model, inputs, labels, loss_fn=torch.nn.CrossEntropyLoss(), lr=1e-4, local_steps=4, use_updates=True, batch_size=0):
    """Take a few gradient descent steps to fit the model to the given input."""
    patched_model = MetaMonkey(model)
    if use_updates:
        patched_model_origin = deepcopy(patched_model)
    for i in range(local_steps):
        if batch_size == 0:
            outputs = patched_model(inputs, patched_model.parameters)
            labels_ = labels
        else:
            idx = i % (inputs.shape[0] // batch_size)
            outputs = patched_model(inputs[idx * batch_size:(idx + 1) * batch_size], patched_model.parameters)
            labels_ = labels[idx * batch_size:(idx + 1) * batch_size]
        loss = loss_fn(outputs, labels_).sum()
        grad = torch.autograd.grad(loss, patched_model.parameters.values())

        patched_model.parameters = OrderedDict((name, param - lr * grad_part)
                                               for ((name, param), grad_part)
                                               in zip(patched_model.parameters.items(), grad))

    if use_updates:
        patched_model.parameters = OrderedDict((name, param - param_origin)
                                               for ((name, param), (name_origin, param_origin))
                                               in zip(patched_model.parameters.items(), patched_model_origin.parameters.items()))
    return list(patched_model.parameters.values())


def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal'):
    # gradients: dummy grad,  input_gradient gt grad
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    elif indices == 'batch':
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 50)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(input_gradient))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(input_gradient))[-50:]
    else:
        raise ValueError()



    ex = list(input_gradient.values())[0]
    # extract the first layer's weights
    if weights == 'linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = list(input_gradient.values())[0].new_ones(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
        for i in indices:
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - list(input_gradient.values())[i]).pow(2)).sum() * weights[i]
            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - list(input_gradient.values())[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - list(input_gradient.values())[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                costs -= (trial_gradient[i] * list(input_gradient.values())[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += list(input_gradient.values())[i].pow(2).sum() * weights[i]
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                   list(input_gradient.values())[i].flatten(),
                                                                   0, 1e-10) * weights[i]
        if cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)