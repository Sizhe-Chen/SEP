'''vanilla training'''
from pickle import NONE
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
import torchvision
import datetime
from utils import *
import os
from sklearn.decomposition import PCA
from robustbench.utils import load_model
import sys
import pynvml
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description='vanilla training')
    parser.add_argument('--target_batch', type=int, default=0)
    parser.add_argument('--model', type=str, default='ResNet18')
    parser.add_argument('--lr', type=float, default=0.1) #0.1
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--ct', '-c', action='store_true')
    parser.add_argument('--debug', '-nd', action='store_false')
    parser.add_argument('--logfile', type=str, default='./log/{}.log')
    parser.add_argument('--ckpt', type=str, default='vanlina-0301-cifar10.pth')
    parser.add_argument('--gpuid', type=str, default='0')
    parser.add_argument('--batch', type=int, default=1300)
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--epoch', type=int, default=30) #30
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps', type=int, default=2)
    parser.add_argument('--alpha', type=int, default=1) # 1
    parser.add_argument('--msvrg', type=int, default=15)
    parser.add_argument('--num_model', type=int, default=15)
    #parser.add_argument('--target_class', type=int, default=0)
    
    parser.add_argument('--testloader', action='store_true')
    #parser.add_argument('--wandb', '-w', action='store_true')
    parser.add_argument('--mse', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed) 
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return args

class_dict = {
    0: 0,
    1: 17,
    2: 33,
    3: 45,
    4: 47,
    5: 51,
    6: 57,
    7: 66,
    8: 83,
    9: 99,
}

def get_dataloader(args):
    if args.dataset == 'cifar10':   trainloader, validloader, testloader = cifar10csz(1., args.batch)
    elif args.dataset == 'cifar100':trainloader, validloader, testloader = cifar100csz(1., args.batch)
    return trainloader, testloader


def compute_correct(outputs: torch.tensor, targets: torch.tensor):
    _, predicted = outputs.max(1)
    return predicted.eq(targets).sum().item()

@torch.no_grad()
def get_mean_feature(net, dataloader, num_of_classes):
    features = torch.zeros(50000, 512, device=device)
    labels = torch.zeros(50000, device=device)

    #pynvml.nvmlInit()
    #handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    for batch_idx, (inputs, targets, index) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        net(inputs)
        features[index] += net.features#net(inputs, return_features=True)
        #meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        #print(batch_idx, meminfo.free / (1024 ** 3))
        #features[index] += net.net.feature
        labels[index] += targets
    features_mean = torch.zeros(num_of_classes, 512, device=device)
    for i in range(num_of_classes): features_mean[i] = features[labels == i].mean(0)[None, ...]
    return features_mean

@torch.no_grad()
def valid_test(net, dataloader, mode, inputs_c, f):
    net.eval()
    total_loss = 0
    correct = 0
    total = 0

    #pynvml.nvmlInit()
    #handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    for batch_idx, (inputs, targets, index) in enumerate(dataloader): # index
        if batch_idx != target_batch: continue
        #print(targets)
        inputs, targets = inputs.to(device), targets.to(device)
        #inputs, targets = inputs[targets != target_class], targets[targets != target_class]
        #pred = net(inputs).argmax(1)

        #for i in range(inputs.shape[0]): print(targets[i], '->', pred[i])
        #exit()
        r_targets = []
        for t in targets: r_targets.append(class_dict[int(t)])
        r_targets = torch.tensor(np.array(r_targets), device=device)

        targets_attack = torch.ones(targets.shape, device=device, dtype=targets.dtype) * ((targets + 5) % 10)
        #((np.array(r_targets) + 50) % 100)
        #for t in targets_attack: print(t)
        #exit()
        
        inputs_c = inputs if inputs_c is None else inputs_c.to(device)
        inputs_c.requires_grad = True

        """
        random_index = torch.arange(inputs.shape[0], device=device)
        np.random.shuffle(random_index)
        net(inputs_c[random_index])
        features_random_order = net.net.features.detach()
        net.zero_grad()
        """
        with torch.enable_grad():
            outputs = net(inputs_c)
            loss = criterion(net.features, f[targets_attack]) #+ criterion(net.net.features, features_random_order)
            #loss = criterion(outputs, targets_attack)
            grad = torch.autograd.grad(loss, inputs_c)[0]
            #meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            #print(batch_idx, meminfo.free / (1024 ** 3))
        
        total_loss += loss.item()
        total += targets.size(0)
        correct += compute_correct(outputs, r_targets)
        log.refresh(batch_idx, len(dataloader), mode, 'Loss:{:.3f} | Acc:{:.3f}%'.format(total_loss / (batch_idx + 1), 100. * correct / total))
        sys.stdout.flush()
        return inputs_c.detach().cpu(), grad.detach().cpu(), outputs.argmax(1).cpu(), targets_attack.cpu()
    return 100. * correct / total, total_loss / (batch_idx + 1)


def yield_task_parameters(): 
    #for tb in range(10): yield '--target_batch=%d --eps=16' % (tb)
    for tb in range(10): yield '--target_batch=%d --eps=8' % (tb)


if __name__ == '__main__':
    args = get_args()
    args = update_ckpt(args)
     
    #wandb.init(project="go-" + args.dataset, entity="hezhengbao", group='vanilla', save_code=True, mode='online' if args.wandb else 'disabled', name='vanilla-{}'.format(args.model))
    log = LogProcessBar(args.logfile, args)
    #wandb.config.update(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    num_of_classes = 10

    criterion = nn.MSELoss()#nn.CrossEntropyLoss()
    #trainloader, testloader = get_dataloader(args)
    trainset = ImageFolder(root='data/SubImageNet', 
        transform=transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),]))
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=False, num_workers=4)
    testloader = None
    loader = testloader if args.testloader else trainloader

    #print('==> Building model..')
    #net = get_model(args.model, num_of_classes=num_of_classes, dataset=args.dataset) # num_of_classes
    #net = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf', model_dir='data') ###
    net = resnet18(num_classes=100)
    net = net.to(device)
    
    eps = args.eps / 255
    target_batch = args.target_batch
    num_step = args.epoch
    inputs_c = None
    
    ckpts = ['ckpt/RN18/' + x for x in os.listdir('ckpt/RN18/')]
    #ckpts = [x for x in ckpts if 'e23' in x or 'e47' in x or 'e71' in x or 'e95' in x or 'e119' in x]
    ckpts.sort(key=lambda x: int(x.split('-')[2][1:]))
    #ckpts = list(range(120))

    interval = int(len(ckpts)/args.num_model)
    used_ckpts = []
    indexes = list(range(len(ckpts)))
    dev = 0
    while len(used_ckpts) != args.num_model:
        used_ckpts = ckpts[interval-dev::interval]
        #used_indexes = indexes[interval-dev::interval]
        dev += 1
    #print(used_ckpts, used_indexes); exit()
    #if args.num_model == 120: used_indexes = ['all']
    path = 'samples/' + ('testvimg_' if args.testloader else 'trainvimg_') + str(args.num_model)#+ '_'.join([str(x) for x in used_indexes])
    os.makedirs(path, exist_ok=True)
    path += '/eps%d-%d_iter%d_sp%05d-%05d' % (args.eps, args.alpha, args.epoch, args.target_batch * args.batch, (args.target_batch+1) * args.batch)

    fs = []
    for i in range(num_step):
        grads = []
        outputs_c = []
        for j, ckpt in enumerate(used_ckpts):
            args.ckpt = ckpt
            args = update_ckpt(args)
            net.load_state_dict(torch.load(args.ckpt))

            if len(fs) < len(used_ckpts): fs.append(get_mean_feature(net, loader, num_of_classes))
            
            inputs_c, grad, outputs, targets_attack = valid_test(net, loader, str(i) + '_' + str(args.ckpt.split('-')[2:]), inputs_c, fs[j])
            grads.append(grad)
            outputs_c.append(outputs)
        if not i: inputs_ori = inputs_c.detach().cpu().clone()

        grad_avg = sum(grads) / len(grads)
        output_class = torch.cat([x[None, ...] for x in outputs_c], 0).cpu().numpy() # (ckpt, batch)
        num_prd_zero = np.where(output_class == targets_attack.detach().cpu().numpy(), 1, 0).sum(0)
        print(num_prd_zero.mean())
        inputs_v = inputs_c.detach().clone()
        grad_sum = 0
        for k in range(args.msvrg):
            ind_ckpt = np.random.choice(range(len(used_ckpts)))
            args.ckpt = used_ckpts[ind_ckpt]
            args = update_ckpt(args)
            net.load_state_dict(torch.load(args.ckpt))

            _, grad_v, _, _ = valid_test(net, loader, str(i) + '_' + str(args.ckpt.split('-')[2:]), inputs_v, fs[j])
            grad_sum += grad_v - (grads[ind_ckpt] - grad_avg)
            inputs_v = torch.clamp(torch.min(torch.max(inputs_v - grad_sum.sign() / 255 * args.alpha, inputs_ori - eps), inputs_ori + eps), 0.0, 1.0)
        
        inputs_c = torch.clamp(torch.min(torch.max(inputs_c - grad_sum.sign() / 255 * args.alpha, inputs_ori - eps), inputs_ori + eps), 0.0, 1.0)

    np.save(path + '_ts%.2f.npy' % num_prd_zero.mean(), inputs_c.detach().cpu().numpy())
        