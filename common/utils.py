import os
from torch._C import Value
import wandb
import torch
import pprint
import random
import argparse
import numpy as np
from termcolor import colored


def setup_run(arg_mode='train'):
    args = parse_args(arg_mode=arg_mode)
    pprint(vars(args))
    print()

    torch.set_printoptions(linewidth=100)
    args.num_gpu = set_gpu(args)
    args.device_ids = None if args.gpu == '-1' else list(range(args.num_gpu))
    args.save_path = os.path.join(f'checkpoints/{args.dataset}/', args.extra_dir)
    ensure_path(args.save_path)

    if not args.no_wandb:
        wandb.init(project=f'mil-{args.dataset}-{args.way}w{args.shot}s',
                   config=args,
                   save_code=True,
                   name=args.extra_dir)

    if args.dataset == 'cm16': 
        args.num_class = 2
    elif args.dataset == 'simclr': 
        args.num_class = 1
    else:
        raise ValueError('Unknown Dataset - Specify class count. ')

    return args


def set_gpu(args):
    if args.gpu == '-1':
        gpu_list = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_list = [int(x) for x in args.gpu.split(',')]
        print('use gpu:', gpu_list)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


def compute_accuracy(logits, labels):
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).type(torch.float).mean().item() * 100.

def compute_accuracy_bce(logits, labels, thr=0.5):
    pred = torch.ge(logits, thr).float()
    return (pred == labels).type(torch.float).mean().item() * 100.


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def load_model(model, dir):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(dir)['params']

    if pretrained_dict.keys() == model_dict.keys():  # load from a parallel meta-trained model and all keys match
        #print('all state_dict keys match, loading model from :', dir)
        model.load_state_dict(pretrained_dict)
    else:  
        ''' Works '''
        model.load_state_dict(pretrained_dict,strict=False)

    return model


def set_seed(seed):
    if seed == 0:
        #print('random seed')
        torch.backends.cudnn.benchmark = True
    else:
        #print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def detect_grad_nan(model):
    for param in model.parameters():
        if param.requires_grad:
            if (param.grad != param.grad):
                if param.grad.float().sum() != 0:  # nan detected
                    param.grad.zero_()


def by(s):
    '''
    :param s: str
    :type s: str
    :return: bold face yellow str
    :rtype: str
    '''
    bold = '\033[1m' + f'{s:.3f}' + '\033[0m'
    yellow = colored(bold, 'yellow')
    return yellow

def print_network(net, show_net=False):
    """ Print network definition"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net) if show_net else print("")
    num_params = num_params / 1000000.0
    print("----------------------------")
    print("MODEL: {:.5f}M".format(num_params))
    print("----------------------------")


def parse_args(arg_mode):
    parser = argparse.ArgumentParser(description='Meta MIL')

    ''' about dataset '''
    parser.add_argument('-dataset', type=str, default='cm16',
                        choices=['cm16', 'simclr'])
    parser.add_argument('-data_dir', type=str, default='datasets', help='dir of datasets')
    parser.add_argument('-data_name',type=str, default='cm16',    help='name of dataset',
    choices=['cm16','simclr', 'msi'])
    
    ''' about wsi-bags dataset '''
    parser.add_argument("-model_ext", type=str, default='-simclr-', help='extra name to add when saving results')
    parser.add_argument("-num_feats", type=int, default=512, help='feature dimension of each instance')
    parser.add_argument("-thres",     type=float, default=0.5, help='optimal threshold for class separation')
    
    ''' about simclr '''
    parser.add_argument("-out_dim", type=int, default=256, help='output dimension of the projection head')
    
    ''' about frmil '''
    parser.add_argument("-mag", type=float, default=8.48, help='margin used in the feature loss (cm16)')
    parser.add_argument('-model_name', type=str, default='frmil', choices=['frmil'])

    ''' about PMSA (MAB - Transformer) '''
    parser.add_argument("-n_heads",    type=int, default=1)
    parser.add_argument('-norm',       action='store_true', help='use layer normalization')
    
    ''' about training specs '''
    parser.add_argument('-batch', type=int, default=2, help='auxiliary batch size')
    parser.add_argument('-max_epoch', type=int, default=200, help='max epoch to run (cm16)')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate (cm16)')
    parser.add_argument('-wd', type=float, default=0.0005, help='learning rate')
    parser.add_argument('-gamma', type=float, default=0.05, help='learning rate decay factor')
    parser.add_argument('-milestones', nargs='+', type=int, default=[100], help='milestones for MultiStepLR')
    parser.add_argument('-save_all', action='store_true', help='save models on each epoch')
    parser.add_argument('-use_adam', action='store_true', help='optimizer choice')

    
    ''' about env '''
    parser.add_argument('-gpu', default='0', help='the GPU ids e.g. \"0\", \"0,1\", \"0,1,2\", etc')
    parser.add_argument('-extra_dir', type=str, default='mil_set', help='extra dir name added to checkpoint dir')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-no_wandb', action='store_true', help='not plotting learning curve on wandb',
                        default=arg_mode == 'test')  # train: enable logging / test: disable logging
    args = parser.parse_args()
    
    return args
