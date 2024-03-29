import torch
from collections import defaultdict,namedtuple
import copy
from .Imprint_class import ImprintBlock
from .r_t_f_attack import ImprintAttacker
import random

import logging

log = logging.getLogger(__name__)

# setup = dict(device=torch.device("cpu"), dtype=torch.float)


class attack_cfg_default:
    type = "analytic"
    attack_type = "imprint-readout"
    label_strategy = "random"  # Labels are not actually required for this attack
    normalize_gradients = False
    impl = namedtuple("impl", ["dtype", "mixed_precision", "JIT"])("float", False, "")


class data_cfg_default:
    size = (1_281_167,)
    classes = 1000
    # shape = (3, 224, 224)
    shape = (3, 32, 32)
    normalize = True
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


def r_b_f(local_model, local_grads, secrets, device):
    local_model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    # label_pred = torch.argmin(torch.sum(list(local_grads.values())[-2], dim=-1), dim=-1).detach().reshape((1,))
    setup = dict(device=device, dtype=torch.float)
    rec_machine = ImprintAttacker(local_model, loss_fn, attack_cfg_default, setup)
    queries = [dict(parameters=[p for p in local_model.parameters()], buffers=[b for b in local_model.buffers()])]
    server_payload = dict(queries=queries, data=data_cfg_default)
    shared_data = dict(
        gradients=local_grads,
        buffers=None,
        num_data_points=1,
        labels=torch.zeros(100),
        local_hypeparams=None,
    )
    reconstructed_data, labels_pred, stats = rec_machine.reconstruct(server_payload, shared_data, secrets, dryrun=False)

    return reconstructed_data,labels_pred
