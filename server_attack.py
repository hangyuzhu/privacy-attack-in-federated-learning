import os
import time
from functools import partial

from fleak.server import ServerAttacker
from fleak.client import Client
from fleak.attack.dummy import TorchDummyImage
from fleak.utils.options import get_model_options
from fleak.utils.constants import DATASETS, MODELS, MODE, ATTACKS, STRATEGY
from fleak.data.image_dataset import N_CLASSES, IMAGE_SHAPE, IMAGE_MEAN_GAN, IMAGE_STD_GAN
from fleak.data.dataloader import federated_dataloaders
from fleak.model import ImprintModel
from fleak.model import GGLGenerator
from fleak.utils.save import save_fed_images, save_acc


def main(args):
    clients_per_round = int(args.total_clients * args.C)

    # ======= Prepare client Dataset ========
    partition_method = dict(iid=args.iid,
                            p_type=args.p_type,
                            beta=args.beta,
                            n_classes=args.num_classes_per_client)
    train_loaders, valid_loaders, test_loaders, test_loader = federated_dataloaders(
        dataset=args.dataset,
        base_data_dir=args.base_data_dir,
        normalize=args.normalize,
        data_augment=args.data_augment,
        p_method=partition_method,
        n_parties=args.total_clients,
        valid_prop=args.valid_prop,
        test_prop=args.test_prop,
        batch_size=args.batch_size,
        verbose=False
    )

    # Assume the attacker holds the mean and std of the training data
    # For methods like robbing the fed, rec_batch_size can be arbitrary values
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
    model = get_model_options(args.dataset)[args.model]

    # adding imprint block
    # often for robbing the fed
    if args.imprint:
        model = partial(ImprintModel, base_module=model, input_shape=dummy.input_shape)
        print("\n###### Wrap the model by imprint module ######")

    # ======= Create Attacker ========
    if args.attack == "ggl":
        import torch
        generator = GGLGenerator()
        model_file = os.path.join("saved_models", "ggl_" + args.dataset + ".pth")
        try:
            generator.load_state_dict(torch.load(model_file))
            print("\n###### Pretrained GGL generator has been loaded ######")
        except:
            print("\n###### Untrained GGL generator is employed ######")
    else:
        generator = None
    server = ServerAttacker(global_model=model(N_CLASSES[args.dataset]),
                            generator=generator,
                            test_loader=test_loader,
                            dummy=dummy,
                            device=args.device)

    # ======= Create Clients ========
    all_clients = [Client(client_id=i,
                          client_model=model(N_CLASSES[args.dataset]),
                          local_epochs=args.local_epochs,
                          lr=args.lr,
                          lr_decay=args.lr_decay,
                          momentum=args.client_momentum,
                          train_loader=train_loaders[i],
                          valid_loader=valid_loaders[i],
                          test_loader=test_loaders[i],
                          device=args.device)
                   for i in range(args.total_clients)]

    # ======= Federated Simulation ========
    eval_accuracy = []

    for i in range(args.num_rounds):
        # check if the communication round is correct or not
        assert i == server.cur_round
        start_time = time.time()
        print('\n====== Round %d of %d: Training %d/%d Clients ======'
              % (i + 1, args.num_rounds, len(all_clients), clients_per_round))
        server.select_clients(online(all_clients), num_clients=min(clients_per_round, len(online(all_clients))))
        eval_acc = server.train_eval(set_to_use=args.set_to_use)
        if i > 0:
            eval_accuracy.append(eval_acc)

        # server side attack
        server.attack(args)

        # federated aggregation
        server.federated_averaging()
        duration_time = time.time() - start_time
        print('One communication round training time: %.4fs' % duration_time)

    # final eval acc
    eval_acc = server.evaluate(set_to_use=args.set_to_use)
    eval_accuracy.append(eval_acc)

    # save
    save_fed_images(dummy, args)
    save_acc(eval_accuracy, args)


def online(clients):
    """We assume all users are always online."""
    return clients


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # training hyperparameters ----------------------------------------------------------------------------
    parser.add_argument('--strategy', type=str, default='fedavg', choices=STRATEGY,
                        help='strategy used in federated learning')
    parser.add_argument('--num_rounds', default=10, type=int, help='num_rounds')
    parser.add_argument('--total_clients', default=10, type=int, help='total number of clients')
    parser.add_argument('--C', default=1, type=float, help='connection ratio')
    parser.add_argument('--local_epochs', default=2, type=int, metavar='N',
                        help='number of local client epochs')
    parser.add_argument('--batch_size', default=50, type=int, metavar='N',
                        help='batch size when training and testing.')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=0.95, type=float, help='learning rate decay')

    parser.add_argument('--client_momentum', default=0.5, type=float, help='learning momentum on client')
    parser.add_argument('--model', default='cnn', type=str, choices=MODELS, help='Training model')
    parser.add_argument('--set_to_use', default='test', type=str, choices=MODE, help='Training model')

    # dataset -----------------------------------------------------------------------------------------------
    parser.add_argument('--base_data_dir', default='../federated_learning/data', type=str,
                        help='base directory of the dataset')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=DATASETS, help='The training dataset')
    parser.add_argument('--data_augment', default=False, action='store_true', help='If using data augmentation')
    parser.add_argument('--normalize', default=False, action='store_true', help='If normalizing data')
    parser.add_argument('--valid_prop', type=float, default=0., help='proportion of validation data')
    parser.add_argument('--test_prop', type=float, default=0.2, help='proportion of test data')
    parser.add_argument('--iid', default=False, action='store_true', help='client dataset partition methods')
    parser.add_argument('--p_type', type=str, default="dirichlet", choices=["dirichlet", "fix_class"],
                        help='type of non-iid partition method')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--num_classes_per_client', type=int, default=2, choices=[2, 5, 20, 50, 100],
                        help='number of data classes on one client')

    # device & save ------------------------------------------------------------------------------------------
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--save_results', default=False, action='store_true', help='if saving the results')

    # attack -------------------------------------------------------------------------------------------------
    parser.add_argument('--attack', default='dlg', type=str, choices=ATTACKS, help='the attack type')
    parser.add_argument('--multi_rec', default=False, action='store_true',
                        help='if using multiple reconstruction from weights')
    parser.add_argument('--imprint', default=False, action='store_true',
                        help='if wrapping the model with imprint block')
    parser.add_argument('--rec_epochs', default=300, type=int, help="reconstruct epochs")
    parser.add_argument('--rec_batch_size', default=1, type=int, metavar='N', help='reconstruction batch size.')
    parser.add_argument('--rec_lr', default=1.0, type=float, help='reconstruct learning rate')

    parser.add_argument('--tv', default=1e-6, type=float, help='hyperparameter for TV regularization')
    # cpa
    parser.add_argument("--decor", type=float, default=1, help="decorrelation weight (CPA)")
    parser.add_argument("--T", type=float, default=5,
                        help="Temperature for cosine similarity when computing decor loss in CPA")
    parser.add_argument("--nv", type=float, default=0, help="negative value penalty")
    parser.add_argument("--l1", type=float, default=0, help="L1 prior")
    parser.add_argument('--fi_lr', default=1e-1, type=float, help='learning rate of feature inversion')
    parser.add_argument("--fi", type=float, default=1, help="feature inversion weight")

    args = parser.parse_args()
    print('\n============== Experimental Settings ==============')
    print(args)
    print('============== Experimental Settings ==============\n')

    main(args)

