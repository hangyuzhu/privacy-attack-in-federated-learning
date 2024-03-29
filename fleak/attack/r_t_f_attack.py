import torch
from collections import defaultdict
import copy
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import logging


log = logging.getLogger(__name__)


class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater thant or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        after_scheduler_dict = {
            key: value for key, value in self.after_scheduler.__dict__.items() if key != "optimizer"
        }
        state_dict = {key: value for key, value in self.__dict__.items() if key != "optimizer"}
        state_dict["after_scheduler"] = after_scheduler_dict
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        after_scheduler_dict = state_dict.pop("after_scheduler")
        self.after_scheduler.__dict__.update(after_scheduler_dict)
        self.__dict__.update(state_dict)


def optimizer_lookup(params, optim_name, step_size, scheduler=None, warmup=0, max_iterations=10_000):
    if optim_name.lower() == "adam":
        optimizer = torch.optim.Adam(params, lr=step_size)
    elif optim_name.lower() == "momgd":
        optimizer = torch.optim.SGD(params, lr=step_size, momentum=0.9, nesterov=True)
    elif optim_name.lower() == "gd":
        optimizer = torch.optim.SGD(params, lr=step_size, momentum=0.0)
    elif optim_name.lower() == "l-bfgs":
        optimizer = torch.optim.LBFGS(params, lr=step_size)
    else:
        raise ValueError(f"Invalid optimizer {optim_name} given.")

    if scheduler == "step-lr":

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[max_iterations // 2.667, max_iterations // 1.6, max_iterations // 1.142], gamma=0.1
        )
    elif scheduler == "cosine-decay":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iterations, eta_min=0.0)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=1)

    if warmup > 0:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=warmup, after_scheduler=scheduler)

    return optimizer, scheduler


class _BaseAttacker:
    """This is a template class for an attack."""

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
        self.cfg = cfg_attack
        self.memory_format = torch.channels_last if cfg_attack.impl.mixed_precision else torch.contiguous_format
        self.setup = dict(device=setup["device"], dtype=getattr(torch, cfg_attack.impl.dtype))
        self.model_template = copy.deepcopy(model)
        self.loss_fn = copy.deepcopy(loss_fn)

    def reconstruct(self, server_payload, shared_data, server_secrets=None, dryrun=False):

        stats = defaultdict(list)

        # Implement the attack here
        # The attack should consume the shared_data and server payloads and reconstruct
        raise NotImplementedError()

        return reconstructed_data, stats

    def __repr__(self):
        raise NotImplementedError()

    def prepare_attack(self, server_payload, shared_data):
        """Basic startup common to many reconstruction methods."""
        stats = defaultdict(list)

        # Load preprocessing constants:
        self.data_shape = server_payload["data"].shape
        self.dm = torch.as_tensor(server_payload["data"].mean, **self.setup)[None, :, None, None]
        self.ds = torch.as_tensor(server_payload["data"].std, **self.setup)[None, :, None, None]

        # Load server_payload into state:
        rec_models = self._construct_models_from_payload_and_buffers(server_payload, shared_data["buffers"])
        shared_data = self._cast_shared_data(shared_data)
        self.rec_models = rec_models
        # Consider label information
        if shared_data["labels"] is None:
            labels = self._recover_label_information(shared_data, rec_models)
        else:
            labels = shared_data["labels"]

        # Condition gradients?
        if self.cfg.normalize_gradients:
            shared_data = self._normalize_gradients(shared_data)
        return rec_models, labels, stats

    def _construct_models_from_payload_and_buffers(self, server_payload, user_buffers):
        """Construct the model (or multiple) that is sent by the server and include user buffers if any."""

        # Load states into multiple models if necessary
        models = []
        for idx, payload in enumerate(server_payload["queries"]):

            new_model = copy.deepcopy(self.model_template)
            new_model.to(**self.setup, memory_format=self.memory_format)

            # Load parameters
            parameters = payload["parameters"]
            if user_buffers is not None and idx < len(user_buffers):
                # User sends buffers. These should be used!
                buffers = user_buffers[idx]
                new_model.eval()
            elif payload["buffers"] is not None:
                # The server has public buffers in any case
                buffers = payload["buffers"]
                new_model.eval()
            else:
                # The user sends no buffers and there are no public bufers
                # (i.e. the user in in training mode and does not send updates)
                new_model.train()
                for module in new_model.modules():
                    if hasattr(module, "track_running_stats"):
                        module.reset_parameters()
                        module.track_running_stats = False
                buffers = []

            with torch.no_grad():
                for param, server_state in zip(new_model.parameters(), parameters):
                    param.copy_(server_state.to(**self.setup))
                for buffer, server_state in zip(new_model.buffers(), buffers):
                    buffer.copy_(server_state.to(**self.setup))

            if self.cfg.impl.JIT == "script":
                example_inputs = self._initialize_data((1, *self.data_shape))
                new_model = torch.jit.script(new_model, example_inputs=[(example_inputs,)])
            elif self.cfg.impl.JIT == "trace":
                example_inputs = self._initialize_data((1, *self.data_shape))
                new_model = torch.jit.trace(new_model, example_inputs=example_inputs)
            models.append(new_model)
        return models

    def _cast_shared_data(self, shared_data):
        """Cast user data to reconstruction data type."""
        cast_grad_list = []
        gradients = copy.deepcopy(shared_data["gradients"])
        grads = tuple(gradients.values())

        # for shared_grad in grads:
        # # for shared_grad in shared_data["gradients"]:
        #     cast_grad_list += [[g.to(dtype=self.setup["dtype"]) for g in shared_grad]]
        cast_grad_list += [[g.to(dtype=self.setup["dtype"]) for g in grads]]
        shared_data["gradients"] = cast_grad_list
        return shared_data

    def _initialize_data(self, data_shape):
        """Note that data is initialized "inside" the network normalization."""
        init_type = self.cfg.init
        if init_type == "randn":
            candidate = torch.randn(data_shape, **self.setup)
        elif init_type == "rand":
            candidate = (torch.rand(data_shape, **self.setup) * 2) - 1.0
        elif init_type == "zeros":
            candidate = torch.zeros(data_shape, **self.setup)
        # Initializations from Wei et al, "A Framework for Evaluating Gradient Leakage
        #                                  Attacks in Federated Learning"
        elif any(c in init_type for c in ["red", "green", "blue", "dark", "light"]):  # init_types like 'red-true'
            candidate = torch.zeros(data_shape, **self.setup)
            if "light" in init_type:
                candidate = torch.ones(data_shape, **self.setup)
            else:
                nonzero_channel = 0 if "red" in init_type else 1 if "green" in init_type else 2
                candidate[:, nonzero_channel, :, :] = 1
            if "-true" in init_type:
                # Shift to be truly RGB, not just normalized RGB
                candidate = (candidate - self.dm) / self.ds
        elif "patterned" in init_type:  # Look for init_type=rand-patterned-4
            pattern_width = int("".join(filter(str.isdigit, init_type)))
            if "rand" in init_type:
                seed = torch.rand([1, 3, pattern_width, pattern_width], **self.setup)
            else:
                seed = torch.rand([1, 3, pattern_width, pattern_width], **self.setup)
            # Shape expansion:
            x_factor, y_factor = (
                torch.as_tensor(data_shape[2] / pattern_width).ceil(),
                torch.as_tensor(data_shape[3] / pattern_width).ceil(),
            )
            candidate = (
                torch.tile(seed, (1, 1, int(x_factor), int(y_factor)))[:, :, : data_shape[2], : data_shape[3]]
                .contiguous()
                .clone()
            )
        else:
            raise ValueError(f"Unknown initialization scheme {init_type} given.")

        candidate.to(memory_format=self.memory_format)
        candidate.requires_grad = True
        candidate.grad = torch.zeros_like(candidate)
        return candidate

    def _init_optimizer(self, candidate):

        optimizer, scheduler = optimizer_lookup(
            [candidate],
            self.cfg.optim.optimizer,
            self.cfg.optim.step_size,
            scheduler=self.cfg.optim.step_size_decay,
            warmup=self.cfg.optim.warmup,
            max_iterations=self.cfg.optim.max_iterations,
        )
        return optimizer, scheduler

    def _normalize_gradients(self, shared_data, fudge_factor=1e-6):
        """Normalize gradients to have norm of 1. No guarantees that this would be a good idea for FL updates."""
        for shared_grad in list(shared_data["gradients"].values()):
            grad_norm = torch.stack([g.pow(2).sum() for g in shared_grad]).sum().sqrt()
            torch._foreach_div_(shared_grad, max(grad_norm, fudge_factor))
        return shared_data

    def _recover_label_information(self, user_data, rec_models):
        """Recover label information.

        This method runs under the assumption that the last two entries in the gradient vector
        correpond to the weight and bias of the last layer (mapping to num_classes).
        For non-classification tasks this has to be modified.

        The behavior with respect to multiple queries is work in progress and subject of debate.
        """
        num_data_points = user_data["num_data_points"]
        num_classes = user_data["gradients"][0][-1].shape[0]
        num_queries = len(user_data["gradients"])

        # In the simplest case, the label can just be inferred from the last layer
        if self.cfg.label_strategy == "iDLG":
            # This was popularized in "iDLG" by Zhao et al., 2020
            # assert num_data_points == 1
            label_list = []
            for query_id, shared_grad in enumerate(user_data["gradients"]):
                last_weight_min = torch.argmin(torch.sum(shared_grad[-2], dim=-1), dim=-1)
                label_list += [last_weight_min.detach()]
            labels = torch.stack(label_list).unique()
        elif self.cfg.label_strategy == "analytic":
            # Analytic recovery simply works as long as all labels are unique.
            label_list = []
            for query_id, shared_grad in enumerate(user_data["gradients"]):
                valid_classes = (shared_grad[-1] < 0).nonzero()
                label_list += [valid_classes]
            labels = torch.stack(label_list).unique()[:num_data_points]
        elif self.cfg.label_strategy == "yin":
            # As seen in Yin et al. 2021, "See Through Gradients: Image Batch Recovery via GradInversion"
            # This additionally assumes that there is a nonlinearity with positive output (like ReLU) in front of the
            # last classification layer.
            # This scheme also works best if all labels are unique
            # Otherwise this is an extension of iDLG to multiple labels:
            total_min_vals = 0
            for query_id, shared_grad in enumerate(user_data["gradients"]):
                total_min_vals += shared_grad[-2].min(dim=-1)[0]
            labels = total_min_vals.argsort()[:num_data_points]

        elif "wainakh" in self.cfg.label_strategy:

            if self.cfg.label_strategy == "wainakh-simple":
                # As seen in Weinakh et al., "User Label Leakage from Gradients in Federated Learning"
                m_impact = 0
                for query_id, shared_grad in enumerate(user_data["gradients"]):
                    g_i = shared_grad[-2].sum(dim=1)
                    m_query = (
                        torch.where(g_i < 0, g_i, torch.zeros_like(g_i)).sum() * (1 + 1 / num_classes) / num_data_points
                    )
                    s_offset = 0
                    m_impact += m_query / num_queries
            elif self.cfg.label_strategy == "wainakh-whitebox":
                # Augment previous strategy with measurements of label impact for dummy data.
                m_impact = 0
                s_offset = torch.zeros(num_classes, **self.setup)

                print("Starting a white-box search for optimal labels. This will take some time.")
                for query_id, (shared_grad, model) in enumerate(zip(user_data["gradients"], rec_models)):
                    # Estimate m:
                    weight_params = (list(rec_models[0].parameters())[-2],)
                    for class_idx in range(num_classes):
                        fake_data = torch.randn([num_data_points, *self.data_shape], **self.setup)
                        fake_labels = torch.as_tensor([class_idx] * num_data_points, **self.setup)
                        with torch.autocast(self.setup["device"].type, enabled=self.cfg.impl.mixed_precision):
                            loss = self.loss_fn(model(fake_data), fake_labels)
                        (W_cls,) = torch.autograd.grad(loss, weight_params)
                        g_i = W_cls.sum(dim=1)
                        m_impact += g_i.sum() * (1 + 1 / num_classes) / num_data_points / num_classes / num_queries

                    # Estimate s:
                    T = num_classes - 1
                    for class_idx in range(num_classes):
                        fake_data = torch.randn([T, *self.data_shape], **self.setup)
                        fake_labels = torch.arange(num_classes, **self.setup)
                        fake_labels = fake_labels[fake_labels != class_idx]
                        with torch.autocast(self.setup["device"].type, enabled=self.cfg.impl.mixed_precision):
                            loss = self.loss_fn(model(fake_data), fake_labels)
                        (W_cls,) = torch.autograd.grad(loss, (weight_params[0][class_idx],))
                        s_offset[class_idx] += W_cls.sum() / T / num_queries

            else:
                raise ValueError(f"Invalid Wainakh strategy {self.cfg.label_strategy}.")

            # After determining impact and offset, run the actual recovery algorithm
            label_list = []
            g_per_query = [shared_grad[-2].sum(dim=1) for shared_grad in user_data["gradients"]]
            g_i = torch.stack(g_per_query).mean(dim=0)
            # Stage 1:
            for idx in range(num_classes):
                if g_i[idx] < 0:
                    label_list.append(torch.as_tensor(idx, device=self.setup["device"]))
                    g_i[idx] -= m_impact
            # Stage 2:
            g_i = g_i - s_offset
            while len(label_list) < num_data_points:
                selected_idx = g_i.argmin()
                label_list.append(torch.as_tensor(selected_idx, device=self.setup["device"]))
                g_i[idx] -= m_impact
            # Finalize labels:
            labels = torch.stack(label_list)

        elif self.cfg.label_strategy == "bias-corrected":  # WIP
            # This is slightly modified analytic label recovery in the style of Wainakh
            bias_per_query = [shared_grad[-1] for shared_grad in user_data["gradients"]]
            label_list = []
            # Stage 1
            average_bias = torch.stack(bias_per_query).mean(dim=0)
            valid_classes = (average_bias < 0).nonzero()
            label_list += [*valid_classes.squeeze(dim=-1)]
            m_impact = average_bias_correct_label = average_bias[valid_classes].sum() / num_data_points

            average_bias[valid_classes] = average_bias[valid_classes] - m_impact
            # Stage 2
            while len(label_list) < num_data_points:
                selected_idx = average_bias.argmin()
                label_list.append(selected_idx)
                average_bias[selected_idx] -= m_impact
            labels = torch.stack(label_list)

        elif self.cfg.label_strategy == "random":
            # A random baseline
            labels = torch.randint(0, num_classes, (num_data_points,), device=self.setup["device"])
        elif self.cfg.label_strategy == "exhaustive":
            # Exhaustive search is possible in principle
            combinations = num_classes ** num_data_points
            raise ValueError(
                f"Exhaustive label searching not implemented. Nothing stops you though from running your"
                f"attack algorithm for any possible combination of labels, except computational effort."
                f"In the given setting, a naive exhaustive strategy would attack {combinations} label vectors."
            )
            # Although this is arguably a worst-case estimate, you might be able to get "close enough" to the actual
            # label vector in much fewer queries, depending on which notion of close-enough makes sense for a given attack.
        else:
            raise ValueError(f"Invalid label recovery strategy {self.cfg.label_strategy} given.")

        # Pad with random labels if too few were produced:
        if len(labels) < num_data_points:
            labels = torch.cat(
                [labels, torch.randint(0, num_classes, (num_data_points - len(labels),), device=self.setup["device"])]
            )

        # Always sort, order does not matter here:
        labels = labels.sort()[0]
        log.info(f"Recovered labels {labels.tolist()} through strategy {self.cfg.label_strategy}.")
        return labels


class AnalyticAttacker(_BaseAttacker):
    """Implements a sanity-check analytic inversion

    Only works for a torch.nn.Sequential model with input-sized FC layers."""

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
        super().__init__(model, loss_fn, cfg_attack, setup)

    def __repr__(self):
        return f"""Attacker (of type {self.__class__.__name__})."""

    def reconstruct(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)

        # Main reconstruction: loop starts here:
        inputs_from_queries = []
        for model, user_gradient in zip(rec_models, list(shared_data["gradients"].values())):
            idx = len(user_gradient) - 1
            for layer in list(model)[::-1]:  # Only for torch.nn.Sequential
                if isinstance(layer, torch.nn.Linear):
                    bias_grad = user_gradient[idx]
                    weight_grad = user_gradient[idx - 1]
                    layer_inputs = self.invert_fc_layer(weight_grad, bias_grad, labels)
                    idx -= 2
                elif isinstance(layer, torch.nn.Flatten):
                    inputs = layer_inputs.reshape(shared_data["num_data_points"], *self.data_shape)
                else:
                    raise ValueError(f"Layer {layer} not supported for this sanity-check attack.")
            inputs_from_queries += [inputs]

        final_reconstruction = torch.stack(inputs_from_queries).mean(dim=0)
        reconstructed_data = dict(data=inputs, labels=labels)

        return reconstructed_data, stats

    def invert_fc_layer(self, weight_grad, bias_grad, image_positions):
        """The basic trick to invert a FC layer."""
        # By the way the labels are exactly at (bias_grad < 0).nonzero() if they are unique
        valid_classes = bias_grad != 0
        intermediates = weight_grad[valid_classes, :] / bias_grad[valid_classes, None]
        if len(image_positions) == 0:
            reconstruction_data = intermediates
        elif len(image_positions) == 1:
            reconstruction_data = intermediates.mean(dim=0)
        else:
            reconstruction_data = intermediates[image_positions]
        return reconstruction_data


class ImprintAttacker(AnalyticAttacker):
    """Abuse imprint secret for near-perfect attack success."""

    def reconstruct(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        """This is somewhat hard-coded for images, but that is not a necessity."""
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)

        if "ImprintBlock" in server_secrets.keys():
            weight_idx = server_secrets["ImprintBlock"]["weight_idx"]
            bias_idx = server_secrets["ImprintBlock"]["bias_idx"]
            data_shape = server_secrets["ImprintBlock"]["shape"]
        else:
            raise ValueError(f"No imprint hidden in model {rec_models[0]} according to server.")

        bias_grad = shared_data["gradients"][0][bias_idx].clone()
        weight_grad = shared_data["gradients"][0][weight_idx].clone()
        # if server_secrets["ImprintBlock"]["structure"] == "cumulative":
        for i in reversed(list(range(1, weight_grad.shape[0]))):
            weight_grad[i] -= weight_grad[i - 1]
            bias_grad[i] -= bias_grad[i - 1]

        image_positions = bias_grad.nonzero()
        layer_inputs = self.invert_fc_layer(weight_grad, bias_grad, [])

        if "decoder" in server_secrets["ImprintBlock"].keys():
            inputs = server_secrets["ImprintBlock"]["decoder"](layer_inputs)
        else:
            inputs = layer_inputs.reshape(layer_inputs.shape[0], *data_shape)[:, :3, :, :]
        if weight_idx > 0:  # An imprint block later in the network:
            inputs = torch.nn.functional.interpolate(
                inputs, size=self.data_shape[1:], mode="bicubic", align_corners=False
            )
        self.dm = self.dm.cuda()
        self.ds = self.ds.cuda()
        inputs = torch.max(torch.min(inputs, (1 - self.dm) / self.ds), -self.dm / self.ds)

        if len(labels) >= inputs.shape[0]:
            # Fill up with zero if not enough data can be found:
            missing_entries = torch.zeros(len(labels) - inputs.shape[0], *self.data_shape, **self.setup)
            inputs = torch.cat([inputs, missing_entries], dim=0)
        else:
            print(f"Initially produced {inputs.shape[0]} hits.")
            # Cut additional hits:
            # this rule is optimal for clean data with few bins:
            # best_guesses = torch.topk(bias_grad[bias_grad != 0].abs(), len(labels), largest=False)
            # this rule is best when faced with differential privacy:
            best_guesses = torch.topk(weight_grad.mean(dim=1)[bias_grad != 0].abs(), len(labels), largest=True)
            print(f"Reduced to {len(labels)} hits.")
            # print(best_guesses.indices.sort().values)
            inputs = inputs[best_guesses.indices]

        # reconstructed_data = dict(data=inputs, labels=labels)
        return inputs, labels, stats