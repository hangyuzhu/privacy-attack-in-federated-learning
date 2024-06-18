import fleak
from fleak.utils.constants import DATASETS, MODELS, ATTACKS


def main(args):
    if args.attack == "dlg" or args.attack == "idlg":
        fleak.dlg_attack(args)
    elif args.attack == "ig_single" or args.attack == "ig_weight" or args.attack == "ig_multi":
        fleak.ig_attack(args)
    elif args.attack == "rtf":
        fleak.rtf_attack(args)
    elif args.attack == "ggl":
        fleak.ggl_attack(args)
    elif args.attack == "grnn":
        fleak.grnn_attack(args)
    elif args.attack == "cpa":
        fleak.cpa_attack(args)
    elif args.attack == "dlf":
        fleak.dlf_attack(args)
    else:
        raise TypeError(f"Unexpected attack method: {args.attack}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--attack', default='dlg', type=str, choices=ATTACKS, help='the attack type')
    # model
    parser.add_argument('--model', default='cnn', type=str, choices=MODELS, help='Training model')
    parser.add_argument('--imprint', default=False, action='store_true',
                        help='if wrapping the model with imprint block')
    # data
    parser.add_argument('--base_data_dir', default='../federated_learning/data', type=str,
                        help='base directory of the dataset')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=DATASETS, help='The training dataset')
    parser.add_argument('--normalize', default=False, action='store_true', help='If normalizing data')
    parser.add_argument('--data_augment', default=False, action='store_true', help='If using data augmentation')
    # device
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--save_results', default=False, action='store_true', help='if saving the results')
    args = parser.parse_args()
    print(args)

    main(args)