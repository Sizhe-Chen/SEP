import sys, os
from collections import OrderedDict
import torch.nn.init as init
import numpy as np
from sklearn.decomposition import PCA

sys.path.append('..')
from models import *
#from advertorch.utils import NormalizeByChannelMeanStd
import datetime
import time


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

class Model(nn.Module):
    def __init__(self, net, data_normalize):
        super(Model, self).__init__()
        self.net = net
        if data_normalize is None:
            self.data_normalize = None
        else:
            self.data_normalize = data_normalize

    def forward(self, x, **kwargs):
        if self.data_normalize is not None:
            x = self.data_normalize(x)
        return self.net(x, **kwargs)


def update_ckpt(args):
    cur_time = datetime.datetime.now().strftime('%y%m%d%H%M%S')
    if 'ckpt' not in args.ckpt:
        args.ckpt = './ckpt/{}/'.format(args.model) + args.ckpt
    if (args.dataset not in args.ckpt) and not args.debug:  # 没有指定dataset的话,且不在debug模式，说明没有指定具体ckpt，则更新为默认格式
        tmp = args.ckpt[:-4].split('/')
        tmp[-1] += '.pth'
        tmp[-1] = '-'.join([cur_time, tmp[-1]])
        args.ckpt = '/'.join(tmp)
        print(args.ckpt)
    args.logfile = args.logfile.format(args.model)
    if not args.debug:
        # args.logfile = '-'.join([args.logfile[:-4], datetime.datetime.now().strftime('%y%m%d%H%M%S'), args.dataset + '.log'])
        args.logfile = args.ckpt.replace('ckpt', 'log').replace(args.model + '/', args.model + '-').replace('.pth', '.log')
    """
    if args.debug: print('-' * 60 + "\n\n\t\t\tWARNING: RUNNING IN DEBUG MODE\n\n" + '-' * 60)
    for key in vars(args): print(key.ljust(8), ' : ', vars(args)[key])
    time.sleep(2)
    """

    return args


def get_model(model_name, num_of_classes=10, dataset=None):
    if dataset == 'cifar10':
        data_normalize = NormalizeByChannelMeanStd(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    elif dataset == 'cifar100':
        data_normalize = NormalizeByChannelMeanStd(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    elif dataset == 'no':
        data_normalize = None
    else:
        raise NotImplementedError

    if model_name == 'ResNet18':
        return Model(ResNet18(num_classes=num_of_classes), data_normalize)
    elif model_name == 'vgg13':
        return Model(VGG('VGG13', num_classes=num_of_classes), data_normalize)
    elif model_name == 'vgg16':
        return Model(VGG('VGG16', num_classes=num_of_classes), data_normalize)
    elif model_name == 'DenseNet':
        return Model(densenet_cifar(num_classes=num_of_classes), data_normalize)
    elif model_name == 'GoogLeNet':
        return Model(GoogLeNet(num_classes=num_of_classes), data_normalize)
    elif model_name == 'SENet18':
        return Model(SENet18(num_classes=num_of_classes), data_normalize)
    elif model_name == 'PreActResNet18':
        return Model(PreActResNet18(num_classes=num_of_classes), data_normalize)
    else:
        raise NotImplementedError


def remove_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
    return new_state_dict


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)
