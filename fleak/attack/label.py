from collections import OrderedDict

import torch


def one_shot_batch_label_restoration(model, gt_grads, dummy_data):
    """One shot batch label restoration

    Towards General Deep Leakage in Federated Learning https://arxiv.org/pdf/2110.09074
    This may cause computation error since p_kn is not the prediction of the real data
    Caution: the total label counts may not be equal to the length of the real data

    :param model: nn.Module
    :param gt_grads: gradients of the ground-truth data
    :param dummy_data: dummy data
    :return: label counts where each slot represents possible occurrences of each class
    """
    K = len(dummy_data)
    # the last layer has bias
    dW = gt_grads[-2].sum(dim=-1)
    z_kn, O_km = model(dummy_data, return_z=True)
    p_kn = torch.softmax(z_kn, dim=-1)

    # prediction sum across sample (k) dimension
    p_n = torch.sum(p_kn, dim=0)
    # the input of the last linear layer sum across feature (m) dimension
    O_k = torch.sum(O_km, dim=-1)
    # approximate O_k
    O_hat = torch.mean(O_k)

    # infer label counts
    label_counts = p_n - K * dW / O_hat
    return label_counts


def label_count_restoration(model,
                            o_state: OrderedDict,
                            n_state: OrderedDict,
                            dummy,
                            local_data_size,
                            epochs,
                            k_batches,
                            batch_size,
                            device):
    """Label count restoration implemented in https://github.com/eth-sri/fedavg_leakage

    We just translate the original code from JAX to PyTorch with slight revisions

    :param model: nn.Module
    :param o_state: original state dict of the model
    :param n_state: updated state dict of the model
    :param dummy: TorchDummy
    :param local_data_size: local data size K
    :param epochs: local training epochs
    :param k_batches: number of batches per epoch
    :param batch_size: batch size
    :param device: cpu or cuda
    :return: label counts where each slot represents possible occurrences of each class
    """
    K = local_data_size
    o_params = [o_state[k] for k, _ in model.named_parameters()]
    # for PyTorch implementation, the size of model weights is (out_dim, in_dim)
    # sum across the input dimension of the last layer weights
    dW = torch.sum(o_params[-2], dim=-1)

    model.load_state_dict(o_state)
    O_start, p_start = calc_label_stats(model, dummy, device)
    model.load_state_dict(n_state)
    O_end, p_end = calc_label_stats(model, dummy, device)

    coefs = torch.arange(0, 1, 1 / (k_batches * batch_size)).to(device)
    O_s = (1 - coefs) * O_start + coefs * O_end
    p_s = (1 - coefs) * p_start + coefs * p_end

    raw_counts = []
    for j in range(0, k_batches * epochs):
        counts = K * p_s[j] - K * dW / O_s[j] / (k_batches * epochs)
        raw_counts.append(counts)
    raw_counts = torch.stack(raw_counts)
    # make the total label counts equal to the local data size
    final_counts = round_label_counts(raw_counts.mean(dim=0), K)
    return final_counts


def calc_label_stats(model, dummy, device):
    k_bs = 1
    k_in = torch.randn([k_bs, *dummy.image_shape], device=device)
    p_k, O_km = model(k_in, return_z=True)
    p_k = torch.mean(torch.softmax(p_k, dim=-1))
    O_k = torch.mean(torch.sum(O_km, dim=-1))
    return O_k, p_k


def round_label_counts(counts, K):
    # Rounding can cause more than input.shape[0] labels. Prevent by taking max
    counts_fl = torch.max(torch.floor(counts).int(), torch.tensor(0))
    counts_rem = K - torch.sum(counts_fl)
    counts_rem_arr = counts - counts_fl
    if counts_rem >= 0:
        _, idx = torch.topk(counts_rem_arr, counts_rem)
        counts_fl[idx] += 1
        counts = counts_fl
    else:
        max_rm = -counts_rem
        rem = 0
        if max_rm > torch.sum(counts_fl >= 0.1):
            rem = max_rm - torch.sum(counts_fl >= 0.1)
            max_rm = torch.sum(counts_fl >= 0.1)

        counts_rem_arr[counts_fl <= 0.1] = 1
        _, idx = torch.topk(-counts_rem_arr, max_rm)
        counts_fl[idx] -= 1
        counts = counts_fl
        if rem > 0:
            _, idx = torch.topk(counts_fl, rem)
            counts[idx] -= 1
    return counts


def label_count_to_label(label_count, device):
    """Convert from label counts to the labels

    Caution: the sequence order is not considered

    :param label_count: label counts where each slot represents possible occurrences of each class
    :param device: cpu or cuda
    :return: integer labels
    """
    labels = []
    for i, c in enumerate(label_count):
        labels.extend([i] * c)
    labels = torch.tensor(labels, device=device)
    return labels
