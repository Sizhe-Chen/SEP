'''vanilla training'''
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
import time
import wandb
from sklearn.metrics import confusion_matrix


def get_args():
    parser = argparse.ArgumentParser(description='vanilla training')
    parser.add_argument('--model', type=str, default='ResNet18')
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--ct', '-c', action='store_true')
    parser.add_argument('--poison', type=float, default=0)
    parser.add_argument('--debug', '-nd', action='store_false')
    parser.add_argument('--logfile', type=str, default='./log/{}.log')
    parser.add_argument('--ckpt', type=str, default='vanillaCIFAR10.pth')
    parser.add_argument('--gpuid', type=str, default='0')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--epoch', type=int, default=120)
    parser.add_argument('--eps', type=str, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--uledir', type=str, default='')
    parser.add_argument('--wandb', '-w', action='store_true')
    args = parser.parse_args()

    if args.uledir != '': args.ckpt = 'uleCIFAR10.pth'
    assert args.dataset == 'cifar10'
    np.random.seed(args.seed)
    torch.manual_seed(args.seed) 
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return args


def get_dataloader(args):
    trainloader, validloader, testloader = cifar10(args.uledir if args.uledir != '' else 1, 
        args.batch, shuffle=[True, False, False], num_workers=args.eps if args.uledir != '' else 4)
    return trainloader, testloader


# Model
def compute_correct(outputs: torch.tensor, targets: torch.tensor):
    _, predicted = outputs.max(1)
    return predicted.eq(targets).sum().item()


def train(epoch, net, trainloader):
    net.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets, index) in enumerate(trainloader):
        if epoch==0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001 + (args.lr-0.001) * (batch_idx+1) /len(trainloader)

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total += targets.size(0)
        correct += compute_correct(outputs, targets)

        log.refresh(batch_idx, len(trainloader), 'Train',
                     'Loss:{:.3f} | Acc:{:.3f}%'.format(total_loss / (batch_idx + 1), 100. * correct / total))
    
    wandb.log({
        'train_loss': total_loss / (batch_idx + 1),
        'train_acc': 100. * correct / total,
    }, step=epoch)


@torch.no_grad()
def valid_test(net, dataloader, mode: str):
    net.eval()
    total_loss = 0
    correct = 0
    total = 0
    correct_per_class = [0 for _ in range(10)]
    totals = [0 for _ in range(10)]
    outputlist = []
    targetlist = []

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)

        outputlist.append(outputs.argmax(1).detach().cpu().numpy())
        targetlist.append(targets.detach().cpu().numpy())

        loss = criterion(outputs, targets)
        total_loss += loss.item()
        total += targets.size(0)
        correct += compute_correct(outputs, targets)
        info = 'Loss:{:.3f} | Acc:{:.3f}%'.format(total_loss / (batch_idx + 1), 100. * correct / total)
        for i in range(10): 
            index = targets == i
            correct_per_class[i] += compute_correct(outputs[index], targets[index])
            totals[i] += index.sum()
            #info += ' | AccC%d:%.3f' % (i, correct_per_class[i] / totals[i] * 100)
        log.refresh(batch_idx, len(dataloader), mode, info)
    cm = confusion_matrix(np.concatenate(targetlist, axis=0), np.concatenate(outputlist, axis=0))
    print(cm)
    
    return 100. * correct / total, total_loss / (batch_idx + 1)


def valid(epoch, net, validloader, args):
    acc, _ = valid_test(net, validloader, 'Valid')
    global best_acc
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        if not os.path.exists(os.path.dirname(args.ckpt)):
            os.mkdir(os.path.dirname(args.ckpt))
        torch.save(state, args.ckpt)
        best_acc = acc


def test(epoch, net, testloader):
    acc, loss = valid_test(net, testloader, 'Test')
    wandb.log({
        'test_clean_loss': loss,
        'test_clean_acc': acc,
    }, step=epoch)
    global best_acc
    if 1:#acc > best_acc:
        #print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        if not os.path.exists(os.path.dirname(args.ckpt)):
            os.mkdir(os.path.dirname(args.ckpt))
        if acc > best_acc: 
            best_acc = acc
            if args.uledir == '': 
                torch.save(state, args.ckpt[:-4] + '-s%d-e%d-a%.2f.pth' % (args.seed, epoch, acc))


def yield_task_parameters(): 
    for uledir in ['samples/traina_1']: # , 10, 20, 60
        for model in ['SENet18', 'vgg16', 'DenseNet', 'GoogLeNet']:
            yield '--uledir=%s --model=%s --eps=8' % (uledir, model)
    

if __name__ == '__main__':
    args = get_args()
    args = update_ckpt(args)

     
    wandb.init(project="go-" + args.dataset, entity="hezhengbao", group='vanilla',
               save_code=True, mode='online' if args.wandb else 'disabled', name='vanilla-{}'.format(args.model))
    log = LogProcessBar(args.logfile, args)

    wandb.config.update(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.dataset == 'cifar10':
        num_of_classes = 10
    elif args.dataset == 'cifar100':
        num_of_classes = 100
    else:
        num_of_classes = 0
        raise NotImplementedError

    print('==> Building model..')
    from robustbench.utils import load_model
    #net = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf', model_dir='data') ###
    net = get_model(args.model, num_of_classes=num_of_classes, dataset=args.dataset)
    net = net.to(device)
    #print(net)
    if device == 'cuda':
        cudnn.benchmark = True

    if args.opt == 'sgd': #pass
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 90], gamma=0.1)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.5)
    else:
        optimizer = None
        raise NotImplementedError

    criterion = nn.CrossEntropyLoss()

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.exists(args.ckpt), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.ckpt)
        print("ckpt folder:", args.ckpt)
        net.load_state_dict(checkpoint['net'])
        if args.ct:
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']
            if 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler'])

    trainloader, testloader = get_dataloader(args)
    
    #net = get_model(args.model, num_of_classes=num_of_classes, dataset=args.dataset)
    #test(0, net, testloader); exit()

    wandb.watch(net, log="all")
    for epoch in range(start_epoch, args.epoch):
        if epoch==1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        log.log('\nEpoch: %d' % epoch)
        print('Epoch: %d' % epoch, 'lr', optimizer.state_dict()['param_groups'][0]['lr'])
        train(epoch, net, trainloader)
        test(epoch, net, testloader)
        scheduler.step()