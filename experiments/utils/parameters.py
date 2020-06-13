import argparse
from .tools import str2bool


def get_parser():
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Pytorch version for "An Efficient PointLSTM for Point Clouds Based Gesture Recognition"(CVPR 2020)')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')
    parser.add_argument(
        '--config',
        default='pointlstm.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--device',
        type=str,
        default=0,
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--phase', default='train', help='must be train or test')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # debug
    parser.add_argument(
        '--random_fix',
        type=str2bool,
        default=True,
        help='fix the random seed or not')
    parser.add_argument(
        '--random-seed',
        type=int,
        default=0,
        help='the default value for random seed.')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=200,
        help='the interval for storing models (#epoch)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=100,
        help='the interval for evaluating models (#epoch)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=20,
        help='the interval for printing messages (#iteration)')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # dataloader
    parser.add_argument(
        '--dataloader', default='dataloader.dataloader', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=4,
        help='the number of workers for data loader')
    parser.add_argument(
        '--framesize',
        type=int,
        default=32,
        help='the number of sampled frame for each video')
    parser.add_argument(
        '--pts-size',
        type=int,
        default=128,
        help='the number of sampled points for each frame')
    parser.add_argument(
        '--train-loader-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-loader-args',
        default=dict(),
        help='the arguments of data loader for testing')
    parser.add_argument(
        '--valid-loader-args',
        default=dict(),
        help='the arguments of data loader for validation')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # optim
    parser.add_argument(
        '--batch-size', type=int, default=16, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=8, help='test batch size')

    default_optimizer_dict = {
        "base_lr": 1e-2,
        "optimizer": "SGD",
        "nesterov": False,
        "step": [100],
        "weight_decay": 0.00005,
        "start_epoch": 0,
    }

    parser.add_argument(
        '--optimizer-args',
        default=default_optimizer_dict,
        help='the arguments of optimizer')

    parser.add_argument(
        '--num-epoch',
        type=int,
        default=200,
        help='stop training in which epoch')
    return parser
