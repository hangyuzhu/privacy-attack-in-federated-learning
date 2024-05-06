import os
import math
import matplotlib.pyplot as plt

from ..attack.dummy import TorchDummy


NUM_COLS = 10


def plot_dummy_images(dummy: TorchDummy, args):
    num_rows = math.ceil(len(dummy.history) / NUM_COLS)

    plt.figure(figsize=(20, 20))
    if len(dummy.labels) == 0:
        # no labels
        for i, _recon in enumerate(dummy.history):
            plt.subplot(num_rows, NUM_COLS, i + 1)
            plt.imshow(_recon)
            plt.axis('off')
    else:
        assert len(dummy.history) == len(dummy.labels)
        for i, (_recon, _label) in enumerate(zip(dummy.history, dummy.labels)):
            plt.subplot(num_rows, NUM_COLS, i + 1)
            plt.imshow(_recon)
            plt.title("l=%d" % _label, fontsize=20)
            plt.axis('off')
    path = r'saved_results'

    if not os.path.exists(path):
        os.makedirs(path)

    if args.iid == True:
        plt.savefig(os.path.join(path, args.attack + args.dataset + str(args.total_clients) + str(args.num_epochs) + str(args.batch_size) + '_fake_image.png'))
    else:
        plt.savefig(os.path.join(path, args.attack + 'noniid' + args.dataset + str(args.total_clients) + str(args.num_epochs) + str(args.batch_size) + '_fake_image.png'))

    plt.show()
