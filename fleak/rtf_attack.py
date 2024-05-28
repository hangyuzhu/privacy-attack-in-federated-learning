import os
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fleak.attack import invert_linear_layer
from fleak.model.imprint import ImprintModel
from fleak.utils.constants import BASE_SAVE_PATH
from fleak.utils.options import get_dataset_options
from fleak.utils.options import get_model_options
from fleak.utils.save import save_images
from fleak.attack.dummy import TorchDummyImage
from fleak.utils.save import NUM_COLS, MAX_IMAGES
from fleak.data.image_dataset import N_CLASSES, IMAGE_SHAPE, IMAGE_MEAN_GAN, IMAGE_STD_GAN


def save_gt_images(images: list, args):
    max_images = min(len(images), MAX_IMAGES)
    num_rows = math.ceil(max_images / NUM_COLS)

    if max_images == 1:
        plt.imshow(images[0])
        plt.axis('off')
    else:
        plt.figure(figsize=(NUM_COLS * 2, num_rows * 2))
        for i in range(max_images):
            plt.subplot(num_rows, NUM_COLS, i + 1)
            plt.imshow(images[i])
            plt.axis('off')

    if args.save_results:
        if not os.path.exists(BASE_SAVE_PATH):
            os.makedirs(BASE_SAVE_PATH)

        filename = f"{BASE_SAVE_PATH}/{args.attack}_gt_{args.dataset}_imp{args.model}_" \
                   f"{args.rec_batch_size}rb.png"
        plt.savefig(filename, bbox_inches='tight')

    plt.show()


def rtf_attack(args):
    assert args.attack == "rtf"
    print(f"\n====== {args.attack} attack ======")

    # attack hyperparameters
    args.num_exp = 1
    args.rec_batch_size = 64
    num_bins = 100
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
    model = ImprintModel(N_CLASSES[args.dataset], model_class, dummy.input_shape, num_bins).to(args.device)

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
        invert_linear_layer(gt_grads, dummy)

    # save
    save_gt_images(images, args)
    save_images(dummy.history, args)