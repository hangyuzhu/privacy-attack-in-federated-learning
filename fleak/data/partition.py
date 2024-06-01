import numpy as np

from .image_dataset import (
    load_mnist_dataset,
    load_cifar10_dataset,
    load_cifar100_dataset,
    load_tiny_imagenet_dataset,
    load_imagenet_dataset
)


def iid_partition(dataset, n_parties):
    """
    :param dataset: torch dataset
    :param n_parties: number of parties
    :return: partitioned data index
    """
    data_idx = np.random.permutation(len(dataset))
    user_idx = np.array_split(data_idx, n_parties)
    return user_idx


def dirichlet_partition(dataset, n_parties, beta):
    """
    :param dataset: torch dataset
    :param n_parties: number of parties
    :param beta: parameter of Dirichlet distribution
    :return: partitioned data index
    """
    y_labels = np.array(dataset.targets)
    n_classes = len(set(y_labels))

    min_samples = 0
    min_required_samples = 15
    user_idx = []
    while min_samples < min_required_samples:
        user_idx = [[] for _ in range(n_parties)]
        for k in range(n_classes):
            data_idx_k = np.where(y_labels == k)[0]
            np.random.shuffle(data_idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = np.array([p * (len(idx) < len(dataset) / n_parties)
                                    for p, idx in zip(proportions, user_idx)])
            proportions = proportions / proportions.sum()
            # the rest of idx would be automatically allocated to the last party
            proportions = (np.cumsum(proportions) * len(data_idx_k)).astype(int)[:-1]
            user_idx = [uidx + idx.tolist() for uidx, idx in zip(user_idx, np.split(data_idx_k, proportions))]
            min_samples = min([len(idx) for idx in user_idx])
    for j in range(n_parties):
        np.random.shuffle(user_idx[j])
    return user_idx


def fix_class_noniid(dataset, num_users, num_classes):
    """
    The partition method proposed by the original FL paper
    :param dataset: torch Dataset
    :param num_users: number of users
    :param num_classes: number of label classes
    :return: partitioned data index
    """
    num_samples_per_client = len(dataset) // num_users
    num_samples_per_class = len(dataset) // 10

    # Default for 10 classes
    class_idx = list(range(10))
    targets = np.array(dataset.targets)

    idx = targets.argsort()
    idxs = {}
    start = 0
    for i in range(10):
        idxs[i] = idx[start:start + num_samples_per_class]
        np.random.shuffle(idxs[i])
        start += num_samples_per_class
    assert len(idxs[0]) == num_samples_per_class

    user_dataset_indexes = {}
    label_users = {}
    removed_idx = []

    for i in range(num_users):
        selected_class = np.random.choice(class_idx, num_classes, replace=False)
        user_dataset_indexes[i] = np.concatenate(
            [idxs[selected_class[j]][0:num_samples_per_client // num_classes]
             for j in range(num_classes)])

        # This is created only for debugging
        label_users[i] = np.concatenate(
            [targets[idxs[selected_class[j]][0:num_samples_per_client // num_classes]]
             for j in range(num_classes)])

        class_idx = list(set(class_idx) - set(selected_class))
        for num in range(num_classes):
            idxs[selected_class[num]] = list(set(idxs[selected_class[num]]) - set(
                idxs[selected_class[num]][0:num_samples_per_client // num_classes]))

            if len(idxs[selected_class[num]]) == 0:
                removed_idx.append(selected_class[num])

        if i != (num_users - 1) and len(class_idx) == 0:
            class_idx = list(set(np.arange(10)) - set(removed_idx))
    print(label_users)
    return user_dataset_indexes


def split_train_valid_test(user_data_indexes, valid_ratio: float, test_ratio: float):
    train_index, valid_index, test_index = {}, {}, {}
    if isinstance(user_data_indexes, dict):
        for c_id, user_data_index in user_data_indexes.items():
            np.random.shuffle(user_data_index)
            len_user_data_index = len(user_data_index)
            valid_size = int(len_user_data_index * valid_ratio)
            test_size = int(len_user_data_index * test_ratio)
            valid_index[c_id] = user_data_index[0:valid_size]
            test_index[c_id] = user_data_index[len_user_data_index - test_size:]
            train_index[c_id] = user_data_index[valid_size:len_user_data_index - test_size]
    elif isinstance(user_data_indexes, list):
        for i, user_data_index in enumerate(user_data_indexes):
            np.random.shuffle(user_data_index)
            len_user_data_index = len(user_data_index)
            valid_size = int(len_user_data_index * valid_ratio)
            test_size = int(len_user_data_index * test_ratio)
            valid_index[i] = user_data_index[0:valid_size]
            test_index[i] = user_data_index[len_user_data_index - test_size:]
            train_index[i] = user_data_index[valid_size:len_user_data_index - test_size]

    else:
        raise TypeError("{} is unexpected data type".format(type(user_data_indexes)))
    return train_index, valid_index, test_index


def partition_dataset(dataset: str, base_data_dir: str, normalize: bool, data_augment: bool, p_method: dict, n_parties,
                      valid_prop=0, test_prop=0.2, verbose=True):
    """
    Training part of the original dataset is allocated to multiple parties, each party manually
    divide the dataset into training / validation / testing data.
    Testing part of the original dataset is not partitioned

    :param dataset: name of the dataset
    :param base_data_dir: base directory of the dataset
    :param normalize: if normalizing the data
    :param data_augment: if using data augmentation
    :param p_method: partition method
    :param n_parties: number of users
    :param valid_prop: proportion of validation data 0 <= v < 1
    :param test_prop: proportion of testing data 0 <= v < 1
    :param verbose: if printing the partitioned client data labels
    :return: train_dataset, test_dataset, train_user_idx, valid_user_idx, test_user_idx
    """
    assert test_prop > 0
    data_dir = f"{base_data_dir}/{dataset}"

    if dataset == 'mnist':
        train_dataset, test_dataset = load_mnist_dataset(data_dir, normalize, data_augment=data_augment)
    elif dataset == 'cifar10':
        train_dataset, test_dataset = load_cifar10_dataset(data_dir, normalize, data_augment=data_augment)
    elif dataset == 'cifar100':
        train_dataset, test_dataset = load_cifar100_dataset(data_dir, normalize, data_augment=data_augment)
    elif dataset == 'tiny_imagenet':
        train_dataset, test_dataset = load_tiny_imagenet_dataset(data_dir, normalize, data_augment=data_augment)
    elif dataset == "imagenet":
        train_dataset, test_dataset = load_imagenet_dataset(data_dir, normalize, data_augment=data_augment)
    else:
        raise TypeError('{} is not an expected dataset !'.format(dataset))

    assert p_method["iid"] in [True, False]
    if p_method["iid"]:
        user_idx = iid_partition(train_dataset, n_parties)
    else:
        assert p_method["p_type"] in ["dirichlet", "fix_class"]
        if p_method["p_type"] == "dirichlet":
            user_idx = dirichlet_partition(train_dataset, n_parties, p_method["beta"])
        else:
            user_idx = fix_class_noniid(train_dataset, n_parties, p_method["n_classes"])
    train_user_idx, valid_user_idx, test_user_idx = split_train_valid_test(user_idx, valid_prop, test_prop)

    if verbose:
        targets = np.array(train_dataset.targets)
        train_user_label = {user: targets[idx] for (user, idx) in train_user_idx.items()}
        valid_user_label = {user: targets[idx] for (user, idx) in valid_user_idx.items()}
        test_user_label = {user: targets[idx] for (user, idx) in test_user_idx.items()}
        print('training labels: ', train_user_label)
        print('validation labels: ', valid_user_label)
        print('testing labels: ', test_user_label)

    return train_dataset, test_dataset, train_user_idx, valid_user_idx, test_user_idx
