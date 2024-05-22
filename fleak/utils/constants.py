DATASETS = ['mnist', 'cifar10', 'cifar100', 'tiny_imagenet']

MODELS = ['mlp', 'cnn', 'simple_cnn',
          'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
          'fc2', 'vgg']
MODE = ['test', 'valid']
STRATEGY = ['fedavg', 'fedper', 'fedcrowd', 'fedbcc', 'bcc']
RESULTS = ['results_fedavg', 'results_fedcrowd', 'results_fedbcc', 'results_bcc']

ATTACKS = ['dlg', 'idlg', 'ig_single', 'ig_multi', 'rtf', 'ggl', 'grnn', 'cpa']


def get_model_options(dataset):
    from ..model import MnistMLP, CifarMLP, MnistConvNet, CifarConvNet, \
        ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, FC2, TinyImageNetVGG, CifarVGG
    model = {
        "mlp": MnistMLP if dataset == 'mnist' else CifarMLP,
        "cnn": MnistConvNet if dataset == 'mnist' else CifarConvNet,
        "resnet18": ResNet18,
        "resnet34": ResNet34,
        "resnet50": ResNet50,
        "resnet101": ResNet101,
        "resnet152": ResNet152,
        "fc2": FC2,
        "vgg": TinyImageNetVGG if dataset == 'tiny_imagenet' else CifarVGG
    }
    return model
