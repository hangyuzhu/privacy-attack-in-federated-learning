import os
import json
import math
import matplotlib.pyplot as plt

from .constants import BASE_SAVE_PATH
from ..attack.dummy import TorchDummy


NUM_COLS = 10
MAX_IMAGES = 100


def save_fed_images(dummy: TorchDummy, args):
    max_images = min(len(dummy.history), MAX_IMAGES)
    num_rows = math.ceil(max_images / NUM_COLS)

    plt.figure(figsize=(NUM_COLS * 2, num_rows * 2))
    if len(dummy.labels) == 0:
        # no labels
        for i in range(max_images):
            plt.subplot(num_rows, NUM_COLS, i + 1)
            plt.imshow(dummy.history[i])
            plt.axis('off')
    else:
        assert len(dummy.history) == len(dummy.labels)
        for i in range(max_images):
            plt.subplot(num_rows, NUM_COLS, i + 1)
            plt.imshow(dummy.history[i])
            plt.title("l=%d" % dummy.labels[i], fontsize=20)
            plt.axis('off')

    if args.save_results:
        if not os.path.exists(BASE_SAVE_PATH):
            os.makedirs(BASE_SAVE_PATH)

        # if normalizing the data
        if args.normalize:
            nz = "nz"
        else:
            nz = "unz"
        # deal with imprint module
        if args.imprint:
            imp = "imp"
        else:
            imp = ""

        if args.iid == True:
            filename = f"{BASE_SAVE_PATH}/{args.strategy}_{args.attack}_{nz}{args.dataset}_{imp}{args.model}_iid_" \
                       f"{args.rec_epochs}re_{args.rec_batch_size}rb_{args.rec_lr}rl_" \
                       f"{args.total_clients}c_{args.num_rounds}r_{args.local_epochs}e_" \
                       f"{args.batch_size}b_{args.lr}l_{args.client_momentum}m.pdf"
        else:
            if args.p_type == "dirichlet":
                filename = f"{BASE_SAVE_PATH}/{args.strategy}_{args.attack}_{nz}{args.dataset}_{imp}{args.model}_" \
                           f"niid{args.beta}_{args.rec_epochs}re_{args.rec_batch_size}rb_{args.rec_lr}rl_" \
                           f"{args.total_clients}c_{args.num_rounds}r_{args.local_epochs}e_" \
                           f"{args.batch_size}b_{args.lr}l_{args.client_momentum}m.pdf"
            else:
                filename = f"{BASE_SAVE_PATH}/{args.strategy}_{args.attack}_{nz}{args.dataset}_{imp}{args.model}_" \
                           f"niid{args.num_classes_per_client}_" \
                           f"{args.rec_epochs}re_{args.rec_batch_size}rb_{args.rec_lr}rl_" \
                           f"{args.total_clients}c_{args.num_rounds}r_{args.local_epochs}e_" \
                           f"{args.batch_size}b_{args.lr}l_{args.client_momentum}m.pdf"
        plt.savefig(filename, bbox_inches='tight')

    # show images
    plt.show()


def save_images(images: list, args):
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

        # if normalizing the data
        if args.normalize:
            nz = "nz"
        else:
            nz = "unz"
        # deal with imprint module
        if args.imprint:
            imp = "imp"
        else:
            imp = ""

        filename = f"{BASE_SAVE_PATH}/{args.attack}_{args.num_exp}n_{nz}{args.dataset}_{imp}{args.model}_" \
                   f"{args.rec_batch_size}rb.pdf"
        plt.savefig(filename, bbox_inches='tight')

    plt.show()


def save_acc(eval_acc: list, args):
    if args.save_results:
        if not os.path.exists(BASE_SAVE_PATH):
            os.makedirs(BASE_SAVE_PATH)

        # if normalizing the data
        if args.normalize:
            nz = "nz"
        else:
            nz = "unz"
        # deal with imprint module
        if args.imprint:
            imp = "imp"
        else:
            imp = ""

        if args.iid == True:
            filename = f"{BASE_SAVE_PATH}/{args.strategy}_{nz}{args.dataset}_{imp}{args.model}_iid_" \
                       f"{args.total_clients}c_{args.num_rounds}r_{args.local_epochs}e_{args.batch_size}b_{args.lr}l_" \
                       f"{args.client_momentum}m.txt"
        else:
            if args.p_type == "dirichlet":
                filename = f"{BASE_SAVE_PATH}/{args.strategy}_{nz}{args.dataset}_{imp}{args.model}_" \
                           f"niid{args.beta}_{args.total_clients}c_{args.num_rounds}r_" \
                           f"{args.local_epochs}e_{args.batch_size}b_{args.lr}l_{args.client_momentum}m.txt"
            else:
                filename = f"{BASE_SAVE_PATH}/{args.strategy}_{nz}{args.dataset}_{imp}{args.model}_" \
                           f"niid{args.num_classes_per_client}_{args.total_clients}c_{args.num_rounds}r_" \
                           f"{args.local_epochs}e_{args.batch_size}b_{args.lr}l_{args.client_momentum}m.txt"
        with open(filename, 'w') as file:
            json.dump(eval_acc, file)