import torch
from torch import optim
import torch.nn as nn

from torchvision.models import *
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys
import numpy as np

import os
import argparse
import wandb
from tqdm import tqdm

def net_param_diff_norm(model:torch.nn.Module, state_dict_init, p='fro'):
    diff_norm_list = []
    for name, parameter in model.named_parameters():
        diff_norm_list.append(torch.norm(parameter.data - state_dict_init[name], p=p).cpu().numpy())
    diff_norm = np.linalg.norm(np.array(diff_norm_list))
    return diff_norm

class_dict = {
    0:  0,
    17: 1,
    33: 2,
    45: 3,
    47: 4,
    51: 5,
    57: 6,
    66: 7,
    83: 8,
    99: 9,
}

path = 'samples/trainvimg_15'
ls = [path + '/' + x for x in os.listdir(path) if 'eps8' in x]
ls.sort()
seps = [torch.tensor(np.load(x)).cuda() for x in ls]
print('loaded', 13000, 'ULE data from', ls)

def step(model,data_loader,opt=None,atk_method=None,*args):
    if opt: model.train()
    else:   model.eval()

    total_loss, total_error = 0., 0.
    total_param_grad_norm = 0.
    total_item = 0

    for X, y, index in tqdm(data_loader):
        X, y = X.cuda(), y.cuda()

        if opt:
            for i in range(X.shape[0]):
                if int(y[i]) not in class_dict: continue
                #print(y[i], int(index[i] / 1300))
                X[i] = seps[class_dict[int(y[i])]][int(index[i] % 1300)]
            """
            si = 0
            for ci in class_dict: si += y == ci
            X, y = X[si], y[si]
            """
        else:
            si = 0
            for ci in class_dict: si += y == ci
            X, y = X[si.bool()], y[si.bool()]

        if X.shape[0]:
            total_item += X.shape[0]
            if atk_method: delta = atk_method(model, X, y, *args)
            else: delta = 0.

            y_pred = model(X + delta)
            #print(type(y_pred))
            #print(y_pred[0])
            try: loss = nn.CrossEntropyLoss()(y_pred, y)
            except TypeError: loss = nn.CrossEntropyLoss()(y_pred[0], y)

            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()

                """
                diff_norm_list = []
                for name, parameter in model.named_parameters():
                    diff_norm_list.append(torch.norm(parameter.grad, p='fro').cpu().numpy())
                diff_norm = np.linalg.norm(np.array(diff_norm_list))
                total_param_grad_norm += diff_norm * X.shape[0].
                """
            
            total_loss += loss.item() * X.shape[0]
            try: total_error += (y_pred.max(dim=1)[1] != y).sum().item()
            except AttributeError: total_error += (y_pred[0].max(dim=1)[1] != y).sum().item()

    avg_loss = total_loss / total_item
    avg_acc = 100 - total_error / total_item * 100
    #avg_param_grad_norm = total_param_grad_norm / len(data_loader.dataset)

    return avg_loss, avg_acc#, avg_param_grad_norm


class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SENet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(2048, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #print(out.shape)
        out = F.avg_pool2d(out, 7)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.linear(out)
        return out


def senet18(num_classes=18):
    return SENet(PreActBlock, [2,2,2,2], num_classes=num_classes)


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--wandb_save', action='store_true', default=False)
    parser.add_argument('--project', type=str, default='One-Pixel ImageNet Subset')

    parser.add_argument('--model', default='RN18')#, choices=['VGG11', 'RN18', 'RN50', 'DN121'])
    parser.add_argument('--data_path', type=str, default='data/SubImageNet')
    parser.add_argument('--ops', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128) # 256
    parser.add_argument('--epochs', type=int, default=120) # 100
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--eps', type=int, default=8)
    parser.add_argument('--uledir', type=str, default='samples')
    parser.add_argument('--save_path', type=str, default='ckpt')

    args = parser.parse_args()
    args.save_path += '/' + args.model
    if args.model == 'VGG16': args.lr /= 10

    np.random.seed(args.seed)
    torch.manual_seed(args.seed) 
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.wandb_save:
        os.environ["WANDB_API_KEY"] = '060a6492e5d7c6548294c6a1df4ba33811b1ab19'
        wandb.init(project=args.project, entity='cycho', name='{}-{}'.format('OPS' if args.ops else 'Clean',args.model))


    '''data preparation'''
    trans = {
            'train': transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.ToTensor(), 
                                        ]),
            'val': transforms.Compose([
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    ])
            }

    train_path = os.path.join(args.data_path, 'train')
    ops_path = os.path.join(args.data_path, 'train_ops')
    val_path = os.path.join(args.data_path, 'val')

    clean_train_data = ImageFolder(root=train_path, transform=trans['train'])
    #ops_train_data = ImageFolder(root=ops_path, transform=trans['val'])
    clean_val_data = ImageFolder(root=val_path, transform=trans['val'])
    
    clean_train_data.targets = np.array(clean_train_data.targets)
    #ops_train_data.targets = np.array(ops_train_data.targets)
    clean_val_data.targets = np.array(clean_val_data.targets)

    clean_train_loader = DataLoader(clean_train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    #ops_train_loader = DataLoader(ops_train_data, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    clean_val_loader = DataLoader(clean_val_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)


    '''training'''
    print(f"model: {args.model}")
    if args.model == 'VGG16':  Net = vgg16(num_classes=100)
    elif args.model == 'RN18': Net = resnet18(num_classes=100)
    elif args.model == 'RN50': Net = resnet50(num_classes=100)
    elif args.model == 'DN121':Net = densenet121(num_classes=100)
    elif args.model == 'GLNet':Net = googlenet(num_classes=100)
    elif args.model == 'SN18': Net = senet18(num_classes=100)


    Net = Net.cuda()
    Opt = optim.SGD(Net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    LrSch = optim.lr_scheduler.MultiStepLR(Opt, gamma=0.1, milestones=[75, 90]) # [60, 120, 180]

    os.makedirs(args.save_path, exist_ok=True)
    from copy import deepcopy
    state_init = deepcopy(Net.state_dict())

    for n in range(args.epochs):
        if args.ops: train_loss, train_acc = step(Net, ops_train_loader, Opt)
        else: train_loss, train_acc = step(Net, clean_train_loader, Opt)
        val_loss, val_acc = step(Net, clean_val_loader)
        #param_diff_norm = net_param_diff_norm(Net, state_init)
        LrSch.step()

        print(
            'epoch {}\n'.format(n+1), 
            #'param_diff_norm:{:.6f}\t'.format(param_diff_norm),
            #'param_grad_norm:{:.6f}\t'.format(param_grad_norm),
            'normal_train_loss:{:.6f}\t'.format(train_loss), 
            'normal_train_acc:{:.4f}%\t'.format(train_acc), 
            'val_loss:{:.6f}\t'.format(val_loss), 
            'val_acc:{:.4f}%\t'.format(val_acc),
            'lr:{:.6f}\t'.format(Opt.param_groups[0]['lr'])
            )   
        sys.stdout.flush()
        
        if args.wandb_save:
            wandb.log({'normal_training_loss': train_loss,
                    'normal_train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    #'param_diff_norm': param_diff_norm,
                    #'param_grad_norm': param_grad_norm
                    })

        #torch.save(Net.state_dict(), args.save_path + '/vanillaImageNetsub-s%d-e%d-a%.2f.pth' % (args.seed, n, val_acc))
        #torch.save(Net.state_dict(), os.path.join(args.save_path, '{}-{}.pkl'.format('OPS' if args.ops else 'Clean', args.model)))
    #print('model saved')