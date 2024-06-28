import copy
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fleak.attack import dlf
from fleak.attack.label import label_count_restoration
from fleak.attack.label import label_count_to_label
from fleak.utils.options import get_dataset_options
from fleak.utils.options import get_model_options
from fleak.utils.save import save_images
from fleak.attack.dummy import TorchDummyImage
from fleak.data.image_dataset import N_CLASSES, IMAGE_SHAPE, IMAGE_MEAN_GAN, IMAGE_STD_GAN


def get_gt_grads(model, features, labels, epochs, data_size, batch_size, rand_batch, optimizer, criterion):
    k_batches = math.ceil(data_size / batch_size)
    # save the original state
    origin_model = copy.deepcopy(model.state_dict())

    model.train()
    for _ in range(epochs):
        if rand_batch:
            order = torch.randperm(data_size)
            features = features[order]
            labels = labels[order]

        # batch training
        for i in range(k_batches):
            # prepare batch data
            st_idx = i * batch_size
            en_idx = min((i + 1) * batch_size, data_size)
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
    return diffs, origin_model, model.state_dict()


def dlf_attack(args):
    """ Do not switch to model.eval() """
    assert args.attack == "dlf"
    print(f"\n====== {args.attack} attack ======")

    # attack hyperparameters
    args.num_exp = 1
    rec_epochs = 200
    rec_lr = 0.1
    local_lr = 0.004
    epochs = 10

    # set False for the official implementation
    # set True for label restoration
    restore_label = True
    data_size = 50
    # bs 10 or 1
    args.rec_batch_size = 10
    rand_batch = True
    tv = 0.0002
    reg_clip = 10
    reg_reorder = 6.075

    print(f"\n====== Reconstruct {data_size} dummy data ======")

    # ======= Prepare Dataset ========
    dataset_loader = get_dataset_options(args.dataset)
    data_dir = f"{args.base_data_dir}/{args.dataset}"

    train_dataset, test_dataset = dataset_loader(data_dir, args.normalize, data_augment=args.data_augment)
    train_dl = DataLoader(train_dataset, batch_size=data_size, shuffle=True)

    # ======= Dummy =======
    dummy = TorchDummyImage(
        image_shape=IMAGE_SHAPE[args.dataset],
        batch_size=args.rec_batch_size,
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
            epochs=epochs,
            data_size=data_size,
            batch_size=args.rec_batch_size,
            rand_batch=rand_batch,
            optimizer=train_optimizer,
            criterion=criterion
        )
        # reset to the original state
        model.load_state_dict(o_state)

        # ======= Reconstruct label counts =======
        if restore_label:
            label_counts = label_count_restoration(
                model, o_state, n_state, gt_grads, dummy, data_size, epochs, args.rec_batch_size, args.device)

            gt_counts = torch.tensor([torch.sum(gt_y == i) for i in range(N_CLASSES[args.dataset])], device=args.device)
            tar_err = torch.sum(torch.abs(gt_counts - label_counts)) / 2
            print(f"tar_error: {tar_err}")

            targets = label_count_to_label(label_counts, args.device)
        else:
            targets = gt_y

        # ======= Private Attack =======
        dlf(model, gt_grads, dummy, targets, rec_epochs, rec_lr, epochs, local_lr,
            data_size, args.rec_batch_size, tv, reg_clip, reg_reorder, args.device)

    # save
    images += dummy.history
    save_images(images, args)
