import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fleak.attack import ggl
from fleak.model import GGLGenerator
from fleak.utils.options import get_dataset_options
from fleak.utils.options import get_model_options
from fleak.utils.save import save_images
from fleak.attack.dummy import TorchDummyImage
from fleak.data.image_dataset import N_CLASSES, IMAGE_SHAPE, IMAGE_MEAN_GAN, IMAGE_STD_GAN


def ggl_attack(args):
    assert args.attack == "ggl"
    print(f"\n====== {args.attack} attack ======")

    # attack hyperparameters
    args.num_exp = 10
    args.rec_epochs = 25000
    args.rec_batch_size = 1
    pretrained = True
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
        device=args.device,
    )

    # ======= Create Model ========
    model_class = get_model_options(args.dataset)[args.model]
    # be careful about model.train() & model.eval() issue
    model = model_class(N_CLASSES[args.dataset]).to(args.device)
    model.eval()

    # ======= Create Generator ========
    generator = GGLGenerator().to(args.device)
    if pretrained:
        model_file = os.path.join("saved_models", "ggl_" + args.dataset + ".pth")
        # if load fails, run train_ggl_cifar10.py
        generator.load_state_dict(torch.load(model_file))
        print("\n###### Pretrained GGL generator has been loaded ######")
    generator.eval()

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
        ggl(model, generator, gt_grads, dummy, args.rec_epochs, args.device)

    # save
    images += dummy.history
    save_images(images, args)