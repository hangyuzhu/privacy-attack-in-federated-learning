import torch


def invert_linear_layer(gt_grads, dummy):
    """ Retrieve the ground-truth data by inverting gradients of the first linear layer

    Robbing the fed: https://arxiv.org/pdf/2110.13057
    We remove redundant codes presented in https://github.com/lhfowl/robbing_the_fed/tree/main
    Imprint module implementation strictly follows the math formula in the paper
    Note this method does not require the label information

    :param gt_grads: gradients of the ground-truth data
    :param dummy: TorchDummy object
    :return: reconstructed data
    """
    # gradients of weights of the first linear layer
    invert_weights = gt_grads[0]
    # gradients of bias of the first linear layer
    invert_bias = gt_grads[1]

    # increase the number of bins may get better (worse) inverting performance
    # larger bins is equivalent to smaller interval between c_l & c_l+1
    # thus, it is more likely that just a single (no) image can be extracted
    for i in range(0, invert_weights.shape[0] - 1):
        # extract the possible single image within c_l <= h(x) <= c_l+1
        # since inverse CDF of Gaussian / Laplacian distribution ranges from -inf to inf
        # c_i = phi^-1(i/k) also increases monotonically
        invert_weights[i] -= invert_weights[i + 1]
        invert_bias[i] -= invert_bias[i + 1]

    # remove dl / db = 0
    valid_bins = invert_bias != 0
    inverted_data = invert_weights[valid_bins, :] / invert_bias[valid_bins, None]
    # batch size of dummy is meaningless here
    inverted_data = inverted_data.reshape(inverted_data.shape[0], *dummy.image_shape)

    # small trick used in
    inverted_data = torch.max(
        torch.min(inverted_data, (1 - dummy.t_dm) / dummy.t_ds), -dummy.t_dm / dummy.t_ds)
    dummy.append(inverted_data)

    return inverted_data
