DATASETS = ['mnist', 'cifar10', 'cifar100', 'tiny_imagenet', 'imagenet']

MODELS = ['mlp', 'lenet', 'cnn', 'simple_cnn', 'cnn3',
          'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
          'fc2', 'vgg16']
MODE = ['test', 'valid']
STRATEGY = ['fedavg', 'fedper']
RESULTS = ['results_fedavg', 'results_fedcrowd', 'results_fedbcc', 'results_bcc']

ATTACKS = ['dlg', 'idlg', 'ig_single', 'ig_weight', 'ig_multi', 'rtf', 'ggl', 'grnn', 'cpa', 'dlf']

BASE_SAVE_PATH = "saved_results"
