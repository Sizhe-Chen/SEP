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
    parser.add_argument('--batch', type=int, default=5000)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--epoch', type=int, default=30) #30
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps', type=int, default=2)
    parser.add_argument('--alpha', type=int, default=1) # 1
    parser.add_argument('--num_model', type=int, default=5)
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


def get_dataloader(args):
    if args.dataset == 'cifar10':   trainloader, validloader, testloader = cifar10csz(1., args.batch)
    elif args.dataset == 'cifar100':trainloader, validloader, testloader = cifar100(0.9, args.batch)
    return trainloader, testloader


def compute_correct(outputs: torch.tensor, targets: torch.tensor):
    _, predicted = outputs.max(1)
    return predicted.eq(targets).sum().item()


@torch.no_grad()
def valid_test(net, dataloader, mode, inputs_c=None):
    net.eval()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, index) in enumerate(dataloader):
        if batch_idx != target_batch: continue
        inputs, targets = inputs.to(device), targets.to(device)
        #inputs, targets = inputs[targets != target_class], targets[targets != target_class]
        targets_attack = torch.ones(targets.shape, device=device, dtype=targets.dtype) * ((targets + 5) % 10)
        inputs_c = inputs if inputs_c is None else inputs_c.to(device)
        inputs_c.requires_grad = True
        with torch.enable_grad():
            outputs = net(inputs_c)
            loss = criterion(outputs, targets_attack)
            grad = torch.autograd.grad(loss, inputs_c)[0]
        
        total_loss += loss.item()
        total += targets.size(0)
        correct += compute_correct(outputs, targets)
        log.refresh(batch_idx, len(dataloader), mode, 'Loss:{:.3f} | Acc:{:.3f}%'.format(total_loss / (batch_idx + 1), 100. * correct / total))
        sys.stdout.flush()
        print()
        return inputs_c.detach().cpu(), grad.detach().cpu(), outputs.argmax(1).cpu(), targets_attack.cpu()
    return 100. * correct / total, total_loss / (batch_idx + 1)


def yield_task_parameters(): 
    #for num_model in [10, 20, 30, 60]:
        #for tb in range(2): yield '--num_model=%d --target_batch=%d' % (num_model, tb)
    
    #for num_model in [30]:
        #for tb in range(5, 10): yield '--num_model=%d --target_batch=%d --trainloader' % (num_model, tb)
    
    for num_model in [1, 5, 30]:
        for tb in range(10): yield '--num_model=%d --target_batch=%d --trainloader' % (num_model, tb)
    
    #for num_model in [60]:
        #for tb in range(5, 10): yield '--num_model=%d --target_batch=%d --trainloader' % (num_model, tb)


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

    if args.dataset == 'cifar10': num_of_classes = 10
    elif args.dataset == 'cifar100': num_of_classes = 100
    else: raise NotImplementedError

    criterion = nn.CrossEntropyLoss()
    trainloader, testloader = get_dataloader(args)
    loader = testloader if args.testloader else trainloader

    #print('==> Building model..')
    net = get_model(args.model, num_of_classes=num_of_classes, dataset=args.dataset) # num_of_classes
    #net = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf', model_dir='data') ###
    net = net.to(device)
    
    eps = args.eps / 255
    target_batch = args.target_batch
    num_step = args.epoch
    inputs_c = None
    
    ckpts = ['ckpt/ResNet18/' + x for x in os.listdir('ckpt/ResNet18/') if 's2-e' in x]
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
    #print(used_ckpts); exit()
    #if args.num_model == 120: used_indexes = ['all']
    path = 'samples/' + ('test_' if args.testloader else 'train_') + str(args.num_model)#+ '_'.join([str(x) for x in used_indexes])
    os.makedirs(path, exist_ok=True)
    path += '/eps%d-%d_iter%d_sp%05d-%05d' % (args.eps, args.alpha, args.epoch, args.target_batch * args.batch, (args.target_batch+1) * args.batch)

    for i in range(num_step):
        grads = []
        outputs_c = []
        for ckpt in used_ckpts:
            args.ckpt = ckpt
            args = update_ckpt(args)
            net.load_state_dict(torch.load(args.ckpt)['net'])

            inputs_c, grad, outputs, targets_attack = valid_test(net, loader, str(i) + '_' + str(args.ckpt.split('-')[2:]), inputs_c)
            grads.append(grad)
            outputs_c.append(outputs)
        if not i: inputs_ori = inputs_c.detach().cpu().clone()

        grad_avg = sum(grads) / len(grads)
        output_class = torch.cat([x[None, ...] for x in outputs_c], 0).cpu().numpy() # (ckpt, batch)
        num_prd_zero = np.where(output_class == targets_attack.detach().cpu().numpy(), 1, 0).sum(0)

        inputs_c = torch.clamp(torch.min(torch.max(inputs_c - grad_avg.sign() / 255 * args.alpha, inputs_ori - eps), inputs_ori + eps), 0.0, 1.0)
        print(num_prd_zero.mean())

    np.save(path + '_ts%.2f.npy' % num_prd_zero.mean(), inputs_c.detach().cpu().numpy())
        