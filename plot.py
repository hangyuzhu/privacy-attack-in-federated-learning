import matplotlib.pyplot as plt
import os
import json
import glob
from fleak.utils.constants import MODELS, DATASETS


def plot_eval_acc(args):
    file_name = args.model + '_' + args.dataset
    if args.iid:
        file_name += '_iid_*.txt'
    else:
        file_name += '_niid' + str(args.beta) + '_*txt'

    file_contents = {}
    for path in args.paths:
        # Get a list of files in the directory
        files = glob.glob(os.path.join(path, file_name))

        # Sort the files by modification time in descending order
        files.sort(key=os.path.getmtime, reverse=True)

        # Retrieve the most recent file
        most_recent_file = files[0]

        # Read the content of the most recent file
        with open(most_recent_file, "r") as file:
            file_content = json.load(file)
        file_contents[path.split('_')[-1]] = file_content

    for key, value in file_contents.items():
        plt.plot(range(len(value)), value, label=key)
    plt.legend()
    # plt.savefig(plot_file_name + '.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # parser.add_argument('--paths', nargs='+', default=['results_fedbcc', 'results_fedcrowd', 'results_bcc', 'results_fedavg'],
    #                     help='strategy used in federated learning')
    parser.add_argument('--paths', nargs='+',
                        default=['results_fedavg'],
                        help='strategy used in federated learning')

    parser.add_argument('--model', default='cnn', type=str, choices=MODELS, help='Training model')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=DATASETS, help='The training dataset')
    parser.add_argument('--iid', default=False, action='store_true', help='client dataset partition methods')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    args = parser.parse_args()

    plot_eval_acc(args)