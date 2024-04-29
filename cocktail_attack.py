import time
import json
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

from fleak.server.server_cock import ServerCock
from fleak.client import Client
from fleak.utils.constants import get_model_options
from fleak.utils.constants import DATASETS, MODELS, MODE, STRATEGY
from fleak.data.dataloader import generate_dataloaders


setup = dict(device="cuda" if torch.cuda.is_available() else "CPU", dtype=torch.float32)
dm = torch.as_tensor([0.4914672374725342, 0.4822617471218109, 0.4467701315879822], device="cuda" if torch.cuda.is_available() else "CPU", dtype=torch.float32)[None, :, None, None]
ds = torch.as_tensor([0.24703224003314972, 0.24348513782024384, 0.26158785820007324], device="cuda" if torch.cuda.is_available() else "CPU", dtype=torch.float32)[None, :, None, None]


def main(args):
    clients_per_round = int(args.total_clients * args.C)

    # ======= Prepare client Dataset ========
    partition_method = dict(iid=args.iid,
                            p_type=args.p_type,
                            beta=args.beta,
                            n_classes=args.num_classes_per_client)
    if args.dataset != "tiny_imagenet":
        data_dir = args.data_path + args.dataset
    else:
        data_dir = args.data_path + 'tiny-imagenet-200'
    train_loaders, valid_loaders, test_loaders, test_loader = generate_dataloaders(
        dataset=args.dataset,
        data_dir=data_dir,
        data_augment=args.data_augment,
        p_method=partition_method,
        n_parties=args.total_clients,
        valid_prop=args.valid_prop,
        test_prop=args.test_prop,
        batch_size=args.batch_size
    )



    tt = transforms.ToPILImage()
    # ======= Create Model ========
    model = get_model_options(args.dataset)[args.model]

    # ======= Create Server ========
    # server = serverdlg.ServerDLG(global_model=model(n_classes),
    #                              momentum=args.server_momentum,
    #                              device=args.device,
    #                              data_size=shape_img,
    #                              label_size=label_size)
    server = ServerCock(global_model=model(ds=args.dataset,h_dim=args.h_dim),
                        test_loader=test_loader,
                        datasets=args.dataset,
                        batch_size=args.batch_size,
                        fc_index= 0,
                        device=args.device)

    # ======= Create Clients ========
    all_clients = [Client(client_id=i,
                          client_model=model(ds=args.dataset, h_dim=args.h_dim),
                          num_epochs=args.num_epochs,
                          lr=args.lr,
                          lr_decay=args.lr_decay,
                          momentum=args.client_momentum,
                          train_loader=train_loaders[i],
                          valid_loader=valid_loaders[i],
                          test_loader=test_loaders[i],
                          device=args.device)
                   for i in range(args.total_clients)]

    # ======= Federated Simulation ========
    start = time.time()
    eval_accuracy = []
    history = []
    plt.figure(figsize=(12, 8))
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


        #attack
        reconstruct_data = server.random_attack()
        history.append(reconstruct_data.clone().detach())

        """before or after ?"""
        server.federated_averaging()
        duration_time = time.time() - start_time
        print('One communication round training time: %.4fs' % duration_time)

    tp = transforms.Compose([transforms.ToPILImage()])

    ## show reconstructions
    if args.dataset == 'mnist':
        for i, _recon in enumerate(history):
            plt.subplot(10, 10, i + 1)
            _recon.mul(255).add_(0.5).clamp_(0, 255)
            _recon = _recon.to(dtype=torch.uint8)
            plt.imshow(_recon[0].permute(1, 2, 0).cpu(), cmap='gray')
            plt.axis('off')
    else:
        for i, _recon in enumerate(history):
            plt.subplot(10, 10, i + 1)
            if _recon.shape[0]>1:
                counter = 0
                for j in range(_recon.shape[0]):
                    counter += 1
                    plt.subplot(10 * len(history), 10, counter)
                    plt.imshow(tp(_recon[j].cpu()))
                    plt.axis('off')
            else:
                plt.subplot(10, 10, i + 1)
                plt.imshow(tp(_recon[0].cpu()))
                plt.axis('off')
    path = r'saved_results'
    if not os.path.exists(path):
        os.makedirs(path)
    if args.iid == True:
        plt.savefig(os.path.join(path, args.attack + args.dataset + str(args.total_clients) + str(args.num_epochs) + str(args.batch_size) + '_fake_image.png'))
    else:
        plt.savefig(os.path.join(path, args.attack + 'noniid' + args.dataset + str(args.total_clients) + str(args.num_epochs) + str(args.batch_size) + '_fake_image.png'))

    # reconstruct_data = reconstruct_data.detach()
    # reconstruct_data.mul_(ds).add_(dm).clamp_(0, 1)
    # reconstruct_data = reconstruct_data.to(dtype=torch.float32)
    # plt.subplot(3, 10, i + 1)
    # plt.imshow(reconstruct_data[0].permute(1, 2, 0).cpu())
    # plt.title(plt.title("round = %d" % i))
    # history.append(tt(reconstruct_data[0].cpu()))
    plt.show()
    # final eval acc
    eval_acc = server.evaluate(set_to_use=args.set_to_use)
    eval_accuracy.append(eval_acc)


def online(clients):
    """We assume all users are always online."""
    return clients


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--strategy', type=str, default='fedavg', choices=STRATEGY,
                        help='strategy used in federated learning')

    parser.add_argument('--num_rounds', default=50, type=int, help='num_rounds')
    parser.add_argument('--total_clients', default=10, type=int, help='total number of clients')
    parser.add_argument('--C', default=1, type=float, help='connection ratio')
    parser.add_argument('--num_epochs', default=2, type=int, metavar='N',
                        help='number of local client epochs')
    parser.add_argument('--batch_size', default=50, type=int, metavar='N',
                        help='batch size when training and testing.')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=0.95, type=float, help='learning rate decay')

    # for fedper
    parser.add_argument('--num_shared_layers', default=-1, type=int, help='number of shared layers for fedper')

    parser.add_argument('--client_momentum', default=0.5, type=float, help='learning momentum on client')
    parser.add_argument('--model', default='fc2', type=str, choices=MODELS, help='Training model')
    parser.add_argument('--set_to_use', default='test', type=str, choices=MODE, help='Training model')

    parser.add_argument('--data_path', default='../federated_learning/data/',
                        type=str, help='path of the dataset')
    parser.add_argument('--dataset', default='tiny_imagenet', type=str, choices=DATASETS, help='The training dataset')
    parser.add_argument('--h_dim', default=256, type=int, choices=DATASETS, help='hidden dim')
    parser.add_argument('--data_augment', default=False, action='store_true', help='If using data augmentation')
    parser.add_argument('--valid_prop', type=float, default=0., help='proportion of validation data')
    parser.add_argument('--test_prop', type=float, default=0.2, help='proportion of test data')
    parser.add_argument('--iid', default=False, action='store_true', help='client dataset partition methods')
    parser.add_argument('--p_type', type=str, default="dirichlet", choices=["dirichlet", "fix_class"],
                        help='type of non-iid partition method')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--num_classes_per_client', type=int, default=2, choices=[2, 5, 20, 50, 100],
                        help='number of data classes on one client')

    parser.add_argument('--save_results', default=False, action='store_true', help='if saving the results')

    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')

    parser.add_argument('--attack', default='Cocktail', help='the attack type')

    args = parser.parse_args()
    print('\n============== Federated Learning Setting ==============')
    print(args)
    print('============== Federated Learning Setting ==============\n')

    main(args)
