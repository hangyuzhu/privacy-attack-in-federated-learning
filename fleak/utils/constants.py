DATASETS = ['mnist', 'cifar10', 'cifar100', 'tiny_imagenet']

MODELS = ['mlp', 'cnn', 'simple_cnn',
          'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'robin']
MODE = ['test', 'valid']
STRATEGY = ['fedavg', 'fedper', 'fedcrowd', 'fedbcc', 'bcc']
RESULTS = ['results_fedavg', 'results_fedcrowd', 'results_fedbcc', 'results_bcc']

ATTACKS = ['dlg', 'idlg', 'ig_single', 'ig_multi', 'robbing']


def get_model_options(dataset):
    from ..model import MnistMLP, CifarMLP, MnistConvNet, CifarConvNet, \
        ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, RobinFed, FC2
    from ..model import MnistGenerator, MnistDiscriminator
    model = {
        "mlp": MnistMLP if dataset == 'mnist' else CifarMLP,
        "cnn": MnistConvNet if dataset == 'mnist' else CifarConvNet,
        "resnet18": ResNet18,
        "resnet34": ResNet34,
        "resnet50": ResNet50,
        "resnet101": ResNet101,
        "resnet152": ResNet152,
        "robin": RobinFed,
        "fc2": FC2
    }
    return model
