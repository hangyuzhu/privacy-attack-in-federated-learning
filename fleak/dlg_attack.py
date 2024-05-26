import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fleak.attack import dlg
from fleak.utils.options import get_dataset_options
from fleak.utils.options import get_model_options
from fleak.utils.save import save_images
from fleak.attack.dummy import TorchDummyImage
from fleak.utils.constants import DATASETS, MODELS
from fleak.data.image_dataset import N_CLASSES, IMAGE_SHAPE, IMAGE_MEAN_GAN, IMAGE_STD_GAN


def dlg_attack(attack="dlg"):
    assert attack in ["dlg", "idlg"]

    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--num_exp', default=10, type=int,
                        help='number of experiments.')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch size when training and testing.')
    parser.add_argument('--model', default='cnn', type=str, choices=MODELS, help='Training model')

    parser.add_argument('--base_data_dir', default='../federated_learning/data', type=str,
                        help='base directory of the dataset')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=DATASETS, help='The training dataset')
    parser.add_argument('--data_augment', default=False, action='store_true', help='If using data augmentation')

    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--save_results', default=False, action='store_true', help='if saving the results')

    parser.add_argument('--rec_epochs', default=300, type=int, help="reconstruct epochs")
    parser.add_argument('--rec_batch_size', default=1, type=int, metavar='N', help='reconstruction batch size.')
    parser.add_argument('--rec_lr', default=1.0, type=float, help='reconstruct learning rate')

    args = parser.parse_args()
    args.attack = attack
    print(args)

    # ======= Prepare Dataset ========
    dataset_loader = get_dataset_options(args.dataset)
    data_dir = f"{args.base_data_dir}/{args.dataset}"

    train_dataset, test_dataset = dataset_loader(data_dir, data_augment=args.data_augment)
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

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

    images = []
    for i in range(args.num_exp):
        model = model_class(N_CLASSES[args.dataset]).to(args.device)
        print(f"\n====== {attack} attack: {i + 1} of {args.num_exp} ======")

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
        if args.attack == "dlg":
            dlg(model, gt_grads, dummy, args.rec_epochs, args.rec_lr, device=args.device)

    # save
    images += dummy.history
    save_images(images, args)
