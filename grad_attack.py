def main(args):
    if args.dataset == 'mnist':
        train_dataset, test_dataset = load_mnist_dataset(args.data_dir)
    elif dataset == 'cifar10':
        train_dataset, test_dataset = load_cifar10_dataset(data_dir, data_augment=data_augment)
    elif dataset == 'cifar100':
        train_dataset, test_dataset = load_cifar100_dataset(data_dir, data_augment=data_augment)
    elif dataset == 'tiny_imagenet':
        train_dataset, test_dataset = load_tiny_imagenet_dataset(data_dir)
    else:
        raise TypeError('{} is not an expected dataset !'.format(dataset))


if __name__ == "__main__":
    main(args)