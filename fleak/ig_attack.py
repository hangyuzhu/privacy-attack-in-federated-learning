import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fleak.attack import ig_single
from fleak.attack import ig_multi
from fleak.attack.ig import multi_step_gradients
from fleak.utils.options import get_dataset_options
from fleak.utils.options import get_model_options
from fleak.utils.save import save_images
from fleak.attack.dummy import TorchDummyImage
from fleak.data.image_dataset import N_CLASSES, IMAGE_SHAPE, IMAGE_MEAN_GAN, IMAGE_STD_GAN


def ig_attack(args):
    if args.attack == "ig_single":
        ig_single_attack(args)
    elif args.attack == "ig_weight":
        ig_weight_attack(args)
    elif args.attack == "ig_multi":
        ig_multi_attack(args)
    else:
        raise ValueError(f"Unexpected attack method {args.attack}")


def ig_single_attack(args):
    assert args.attack == "ig_single"
    print(f"\n====== {args.attack} attack ======")

    # attack hyperparameters
    args.num_exp = 10
    args.rec_epochs = 4000
    args.rec_batch_size = 1
    args.rec_lr = 0.1
    tv = 1e-6
    print(f"\n====== Reconstruct {args.rec_batch_size} dummy data ======")

    # ======= Prepare Dataset ========
    dataset_loader = get_dataset_options(args.dataset)
    data_dir = f"{args.base_data_dir}/{args.dataset}"

    train_dataset, test_dataset = dataset_loader(data_dir, data_augment=args.data_augment)
    test_dl = DataLoader(test_dataset, batch_size=args.rec_batch_size, shuffle=True)

    # ======= Dummy =======
    dummy = TorchDummyImage(
        image_shape=IMAGE_SHAPE[args.dataset],
        batch_size=args.rec_batch_size,
        n_classes=N_CLASSES[args.dataset],
        dm=IMAGE_MEAN_GAN[args.dataset],
        ds=IMAGE_STD_GAN[args.dataset],
        device=args.device,
    )

    # ======= Create Model ========
    model_class = get_model_options(args.dataset)[args.model]
    # be careful about model.train() & model.eval() issue
    model = model_class(N_CLASSES[args.dataset]).to(args.device)
    # similar to the official implementation
    model.eval()

    images = []
    for i in range(args.num_exp):
        print(f"\n====== {args.attack} attack: {i + 1} of {args.num_exp} ======")

        # ======= Sample Ground-truth Data ========
        gt_x, gt_y = next(iter(test_dl))
        # restore ground-truth images
        images.extend(dummy.invert_transform(x) for x in gt_x)
        gt_x, gt_y = gt_x.to(args.device), gt_y.to(args.device)

        # ======= Collect Gradients of the Ground-truth Data ========
        criterion = nn.CrossEntropyLoss().to(args.device)
        pred = model(gt_x)
        # assume true labels are achievable
        # however, this is a very strong assumption
        loss = criterion(pred, gt_y)
        gt_grads = torch.autograd.grad(loss, model.parameters())
        gt_grads = [g.detach() for g in gt_grads]

        # ======= Private Attack =======
        ig_single(model, gt_grads, dummy, args.rec_epochs, args.rec_lr, tv, device=args.device)

    # save
    images += dummy.history
    save_images(images, args)


def ig_weight_attack(args):
    assert args.attack == "ig_weight"
    print(f"\n====== {args.attack} attack ======")

    # attack hyperparameters
    args.num_exp = 10
    args.rec_epochs = 8000
    args.rec_batch_size = 1
    args.rec_lr = 0.1
    local_epochs = 5    # training steps
    local_lr = 1e-4     # learning rate
    tv = 1e-6
    print(f"\n====== Reconstruct {args.rec_batch_size} dummy data ======")

    # ======= Prepare Dataset ========
    dataset_loader = get_dataset_options(args.dataset)
    data_dir = f"{args.base_data_dir}/{args.dataset}"

    train_dataset, test_dataset = dataset_loader(data_dir, data_augment=args.data_augment)
    test_dl = DataLoader(test_dataset, batch_size=args.rec_batch_size, shuffle=True)

    # ======= Dummy =======
    dummy = TorchDummyImage(
        image_shape=IMAGE_SHAPE[args.dataset],
        batch_size=args.rec_batch_size,
        n_classes=N_CLASSES[args.dataset],
        dm=IMAGE_MEAN_GAN[args.dataset],
        ds=IMAGE_STD_GAN[args.dataset],
        device=args.device,
    )

    # ======= Create Model ========
    model_class = get_model_options(args.dataset)[args.model]
    # be careful about model.train() & model.eval() issue
    model = model_class(N_CLASSES[args.dataset]).to(args.device)
    # similar to the official implementation
    model.eval()

    images = []
    for i in range(args.num_exp):
        print(f"\n====== {args.attack} attack: {i + 1} of {args.num_exp} ======")

        # ======= Sample Ground-truth Data ========
        gt_x, gt_y = next(iter(test_dl))
        # restore ground-truth images
        images.extend(dummy.invert_transform(x) for x in gt_x)
        gt_x, gt_y = gt_x.to(args.device), gt_y.to(args.device)

        # ======= Collect Gradients (Differences) of the Ground-truth Data ========
        criterion = nn.CrossEntropyLoss().to(args.device)
        model.zero_grad()
        gt_grads = multi_step_gradients(model, gt_x, gt_y, criterion, local_epochs, local_lr)
        gt_grads = [g.detach() for g in gt_grads]

        # ======= Private Attack =======
        ig_multi(model, gt_grads, dummy, args.rec_epochs, args.rec_lr, local_epochs, local_lr, tv, args.device)

    # save
    images += dummy.history
    save_images(images, args)


def ig_multi_attack(args):
    assert args.attack == "ig_multi"
    print(f"\n====== {args.attack} attack ======")

    # attack hyperparameters
    args.num_exp = 1
    args.rec_epochs = 24000
    args.rec_batch_size = 10
    args.rec_lr = 1
    local_epochs = 5    # training steps
    local_lr = 1e-4     # learning rate
    tv = 1e-6
    print(f"\n====== Reconstruct {args.rec_batch_size} dummy data ======")

    # ======= Prepare Dataset ========
    dataset_loader = get_dataset_options(args.dataset)
    data_dir = f"{args.base_data_dir}/{args.dataset}"

    train_dataset, test_dataset = dataset_loader(data_dir, data_augment=args.data_augment)
    test_dl = DataLoader(test_dataset, batch_size=args.rec_batch_size, shuffle=True)

    # ======= Dummy =======
    dummy = TorchDummyImage(
        image_shape=IMAGE_SHAPE[args.dataset],
        batch_size=args.rec_batch_size,
        n_classes=N_CLASSES[args.dataset],
        dm=IMAGE_MEAN_GAN[args.dataset],
        ds=IMAGE_STD_GAN[args.dataset],
        device=args.device,
    )

    # ======= Create Model ========
    model_class = get_model_options(args.dataset)[args.model]
    # be careful about model.train() & model.eval() issue
    model = model_class(N_CLASSES[args.dataset]).to(args.device)
    # similar to the official implementation
    model.eval()

    images = []
    for i in range(args.num_exp):
        print(f"\n====== {args.attack} attack: {i + 1} of {args.num_exp} ======")

        # ======= Sample Ground-truth Data ========
        gt_x, gt_y = next(iter(test_dl))
        # restore ground-truth images
        images.extend(dummy.invert_transform(x) for x in gt_x)
        gt_x, gt_y = gt_x.to(args.device), gt_y.to(args.device)

        # ======= Collect Gradients (Differences) of the Ground-truth Data ========
        criterion = nn.CrossEntropyLoss().to(args.device)
        model.zero_grad()
        gt_grads = multi_step_gradients(model, gt_x, gt_y, criterion, local_epochs, local_lr)
        gt_grads = [g.detach() for g in gt_grads]

        # ======= Private Attack =======
        ig_multi(model, gt_grads, dummy, args.rec_epochs, args.rec_lr, local_epochs, local_lr, tv, args.device)

    # save
    images += dummy.history
    save_images(images, args)