import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from fleak.attack import cpa
from fleak.utils.options import get_dataset_options
from fleak.utils.options import get_model_options
from fleak.utils.save import save_images
from fleak.attack.dummy import TorchDummyImage
from fleak.data.image_dataset import ImageNet
from fleak.data.image_dataset import N_CLASSES, IMAGE_SHAPE, IMAGE_STD_GAN, IMAGE_MEAN_GAN


def cpa_attack(args):
    """ Switch to model.eval() for the period of attack """
    if args.model == "fc2":
        cpa_fc(args)
    elif args.model == "vgg16":
        cpa_fi(args)
    else:
        raise TypeError(f"Unexpected model: {args.model}")


def cpa_fc(args):
    print(f"\n====== {args.attack} fc attack ======")

    # attack hyperparameters
    args.num_exp = 1
    args.rec_batch_size = 50
    rec_epochs = 25000
    rec_lr = 0.001
    fi_lr = 1e-1
    decor = 1.47
    T = 12.4
    tv = 3.1
    nv = 0
    l1 = 0
    fi = 1
    print(f"\n====== Reconstruct {args.rec_batch_size} dummy data ======")

    # ======= Prepare Dataset ========
    dataset_loader = get_dataset_options(args.dataset)
    data_dir = f"{args.base_data_dir}/{args.dataset}"

    train_dataset, test_dataset = dataset_loader(data_dir, args.normalize, data_augment=args.data_augment)
    test_dl = DataLoader(test_dataset, batch_size=args.rec_batch_size, shuffle=True)

    # ======= Dummy =======
    dummy = TorchDummyImage(
        image_shape=IMAGE_SHAPE[args.dataset],
        batch_size=args.rec_batch_size,
        n_classes=N_CLASSES[args.dataset],
        normalize=args.normalize,
        dm=IMAGE_MEAN_GAN[args.dataset],
        ds=IMAGE_STD_GAN[args.dataset],
        device=args.device
    )

    # ======= Create Model ========
    model_class = get_model_options(args.dataset)[args.model]
    # be careful about model.train() & model.eval() issue
    model = model_class(N_CLASSES[args.dataset]).to(args.device)
    model.train()

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
        cpa(model, gt_grads, dummy, rec_epochs, rec_lr, fi_lr, decor, T, tv, nv, l1, fi, args.device)

    # save
    images += dummy.history
    save_images(images, args)


def cpa_fi(args):
    print(f"\n====== {args.attack} fi attack ======")

    # attack hyperparameters
    args.pretrained = True

    # pretrained vgg16 model provided by PyTorch
    # you can download pretrained model from https://www.flyai.com/m/vgg16-397923af.pth
    # and save it within the created folder saved_models
    model_file = "saved_models/vgg16-397923af.pth"

    args.num_exp = 1
    args.rec_batch_size = 30
    rec_epochs = 25000
    rec_lr = 1e-3
    fi_lr = 1e-1
    decor = 5.3
    T = 7.7
    tv = 0.1
    nv = 0.13
    l1 = 5
    fi = 1
    print(f"\n====== Reconstruct {args.rec_batch_size} dummy data ======")

    # ======= Prepare ImageNet ========
    data_dir = f"{args.base_data_dir}/{args.dataset}"
    transform_eval_list = [
        transforms.ToTensor(),
        transforms.Resize(size=(256, 256)),
        transforms.CenterCrop(size=(224, 224))
    ]
    if args.normalize:
        dm, ds = IMAGE_MEAN_GAN["imagenet"], IMAGE_STD_GAN["imagenet"]
        transform_eval_list += [transforms.Normalize(dm, ds)]
    transform_eval = transforms.Compose(transform_eval_list)
    test_dataset = ImageNet(data_dir, train=False, transform=transform_eval)
    test_dl = DataLoader(test_dataset, batch_size=args.rec_batch_size, shuffle=True)

    # ======= Dummy =======
    dummy = TorchDummyImage(
        image_shape=IMAGE_SHAPE[args.dataset],
        batch_size=args.rec_batch_size,
        n_classes=N_CLASSES[args.dataset],
        normalize=args.normalize,
        dm=IMAGE_MEAN_GAN[args.dataset],
        ds=IMAGE_STD_GAN[args.dataset],
        device=args.device
    )

    # ======= Create Model ========
    model_class = get_model_options(args.dataset)[args.model]
    # be careful about model.train() & model.eval() issue
    model = model_class(N_CLASSES[args.dataset]).to(args.device)
    if args.pretrained:
        model.load_state_dict(torch.load(model_file))
    model.train()

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
        cpa(model, gt_grads, dummy, rec_epochs, rec_lr, fi_lr, decor, T, tv, nv, l1, fi, args.device)

    # save
    images += dummy.history
    save_images(images, args)