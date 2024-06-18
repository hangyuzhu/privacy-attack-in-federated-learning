import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fleak.attack import dlf
from fleak.attack import label_count_restoration
from fleak.attack.label import label_count_to_label
from fleak.utils.options import get_dataset_options
from fleak.utils.options import get_model_options
from fleak.utils.save import save_images
from fleak.attack.dummy import TorchDummyImage
from fleak.data.image_dataset import N_CLASSES, IMAGE_SHAPE, IMAGE_MEAN_GAN, IMAGE_STD_GAN


def get_gt_grads(model, features, labels, epochs, k_batches, batch_size, rand_batch, optimizer, criterion):
    data_len = len(features)
    assert data_len == k_batches * batch_size

    # save the original state
    origin_model = copy.deepcopy(model.state_dict())

    model.train()
    for _ in range(epochs):
        if rand_batch:
            order = torch.randperm(data_len)
            features = features[order]
            labels = labels[order]

        # batch training
        for i in range(k_batches):
            # prepare batch data
            st_idx = i * batch_size
            en_idx = min((i + 1) * batch_size, data_len)
            x_batch = features[st_idx:en_idx]
            y_batch = labels[st_idx:en_idx]

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    # get the gradients for multiple steps
    lr = optimizer.param_groups[0]["lr"]
    diffs = [((origin_model[k] - v) / lr).detach() for k, v in model.named_parameters()]

    # retrieve to untrained state
    # model.load_state_dict(origin_model)
    return diffs, origin_model, model.state_dict()


def dlf_attack(args):
    """ Do not switch to model.eval() """
    assert args.attack == "dlf"
    print(f"\n====== {args.attack} attack ======")

    # attack hyperparameters
    args.num_exp = 1
    args.epochs = 10
    args.rec_epochs = 200

    local_lr = 0.004
    args.rec_lr = 0.1

    restore_label = True
    # 5 for bs 10, 50 for bs 1
    args.k_batches = 5
    args.rec_batch_size = 10
    rand_batch = True
    tv = 0.0002
    reg_clip = 10
    reg_reorder = 6.075

    print(f"\n====== Reconstruct {args.k_batches * args.rec_batch_size} dummy data ======")

    # ======= Prepare Dataset ========
    dataset_loader = get_dataset_options(args.dataset)
    data_dir = f"{args.base_data_dir}/{args.dataset}"

    train_dataset, test_dataset = dataset_loader(data_dir, args.normalize, data_augment=args.data_augment)
    train_dl = DataLoader(train_dataset, batch_size=args.rec_batch_size * args.k_batches, shuffle=True)

    # ======= Dummy =======
    dummy = TorchDummyImage(
        image_shape=IMAGE_SHAPE[args.dataset],
        batch_size=args.k_batches * args.rec_batch_size,
        n_classes=N_CLASSES[args.dataset],
        normalize=args.normalize,
        dm=IMAGE_MEAN_GAN[args.dataset],
        ds=IMAGE_STD_GAN[args.dataset],
        device=args.device,
    )

    # ======= Create Model ========
    model_class = get_model_options(args.dataset)[args.model]
    # be careful about model.train() & model.eval() issue
    model = model_class(N_CLASSES[args.dataset]).to(args.device)

    images = []
    for i in range(args.num_exp):
        print(f"\n====== {args.attack} attack: {i + 1} of {args.num_exp} ======")

        # ======= Sample Ground-truth Data ========
        gt_x, gt_y = next(iter(train_dl))
        # restore ground-truth images
        images.extend(dummy.invert_transform(x) for x in gt_x)

        # ======= Get Gradients of Multiple Steps ========
        gt_x, gt_y = gt_x.to(args.device), gt_y.to(args.device)
        train_optimizer = optim.SGD(model.parameters(), lr=local_lr)
        criterion = nn.CrossEntropyLoss().to(args.device)
        # cannot change model parameters
        gt_grads, o_state, n_state = get_gt_grads(
            model=model,
            features=gt_x,
            labels=gt_y,
            epochs=args.epochs,
            k_batches=args.k_batches,
            batch_size=args.rec_batch_size,
            rand_batch=rand_batch,
            optimizer=train_optimizer,
            criterion=criterion
        )

        # ======= Reconstruct label counts =======
        if restore_label:
            label_counts = label_count_restoration(
                model, o_state, n_state, dummy, len(gt_x), args.epochs, args.k_batches, args.rec_batch_size, args.device)

            gt_counts = torch.tensor([torch.sum(gt_y == i) for i in range(N_CLASSES[args.dataset])], device=args.device)
            tar_err = torch.sum(torch.abs(gt_counts - label_counts)) / 2
            print(f"tar_error: {tar_err}")

            restored_y = label_count_to_label(label_counts, args.device)

        # ======= Private Attack =======
        dlf(model, gt_grads, dummy, restored_y, args.rec_epochs, args.rec_lr, args.epochs, local_lr,
            args.k_batches, args.rec_batch_size, tv, reg_clip, reg_reorder, args.device)

    # save
    images += dummy.history
    save_images(images, args)
    # save_images(dummy.history, args)