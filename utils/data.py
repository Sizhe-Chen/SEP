import os
import torchvision
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch
import time
from copy import deepcopy


def unpickle(filename):
    with open(filename, 'rb') as fo: dict = pickle.load(fo, encoding='bytes')
    return dict


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()  # 会将数据归一化0-1
])
transform_valid = transforms.Compose([
    transforms.ToTensor()  # 会将数据归一化0-1
])


class TrainDatasetCifar10(Dataset):
    def __init__(self):
        datas = [unpickle('data/cifar10/data_batch_{}'.format(i + 1)) for i in range(5)]
        self.data_ori = np.concatenate([i[b'data'] for i in datas], axis=0).reshape(50000, 3, 32, 32)
        self.label_ori = np.array([label for labels in [data[b'labels'] for data in datas] for label in labels])
        #np.random.shuffle(self.label)
        self.data = np.concatenate([i[b'data'] for i in datas], axis=0).reshape(50000, 3, 32, 32)
        self.label = np.array([label for labels in [data[b'labels'] for data in datas] for label in labels])
        """
        for i in range(10):
            self.data[i * 5000 : (i+1) * 5000] = self.data_ori[self.label_ori == i]
            self.label[i * 5000 : (i+1) * 5000] = self.label_ori[self.label_ori == i]
        #"""
        self.transform = transform_valid

    def __getitem__(self, index):
        tmp_data = self.data[index]
        img = Image.fromarray(np.transpose(tmp_data, (1, 2, 0)))
        return self.transform(img), self.label[index], index

    def __len__(self):
        return 50000


class TrainDatasetCifar10ULE(Dataset):
    def __init__(self, path, eps):
        datas = [unpickle('data/cifar10/data_batch_{}'.format(i + 1)) for i in range(5)]
        self.data = np.concatenate([i[b'data'] for i in datas], axis=0).reshape(50000, 3, 32, 32)
        self.label = np.array([label for labels in [data[b'labels'] for data in datas] for label in labels])

        """
        import PIL.Image
        for i in range(20):
            PIL.Image.fromarray((self.data[i].transpose((1, 2, 0))).astype(np.uint8)).resize((224, 224)).save('vis/ori-%02d.png' % i)
        exit()
        """

        ls = [path + '/' + x for x in os.listdir(path) if 'eps%d' % eps in x]
        ls.sort()
        for l in ls: print(l)
        if os.path.exists(path + '/poisoned_training_labels.npy'):
            self.data = np.load(ls[0]).transpose(0, 3, 1, 2)
            print('loaded', self.data.shape[0],'ULE data from', path)

            self.label = np.load(path + '/poisoned_training_labels.npy').astype(np.uint8)
            print('loaded', self.label.shape[0], 'labels from', path + '/poisoned_training_labels.npy')
        else:
            self.data = (np.concatenate([np.load(x) for x in ls], axis=0) * 255).astype(np.uint8)
            print('loaded', self.data.shape[0],'ULE data from', path)
        self.transform = transform_valid

    def __getitem__(self, index):
        tmp_data = self.data[index]
        img = Image.fromarray(np.transpose(tmp_data, (1, 2, 0)))
        return self.transform(img), self.label[index], index

    def __len__(self):
        return self.data.shape[0]


class TestDatasetCifar10(Dataset):
    def __init__(self):
        import pickle
        with open('data/cifar10/test_batch', 'rb') as fo: dict = pickle.load(fo, encoding='bytes')
        self.data = dict[b'data'].reshape(10000, 3, 32, 32)
        self.label = dict[b'labels']
        self.transform = transform_valid

    def __getitem__(self, index):
        tmp_data = self.data[index]
        img = Image.fromarray(np.transpose(tmp_data, (1, 2, 0)))
        return self.transform(img), self.label[index]

    def __len__(self):
        return 10000


class TrainDatasetCifar100ULE(Dataset):
    def __init__(self, path, eps):
        dict = unpickle('data/cifar100/train')
        self.data = np.array(dict[b'data'].reshape(50000, 3, 32, 32))
        self.label = np.array(dict[b'fine_labels'])

        ls = [path + '/' + x for x in os.listdir(path) if 'eps%d' % eps in x]
        ls.sort()
        for l in ls: print(l)
        if os.path.exists(path + '/poisoned_training_labels.npy'):
            self.data = np.load(ls[0]).transpose(0, 3, 1, 2)
            print('loaded', self.data.shape[0],'ULE data from', path)

            self.label = np.load(path + '/poisoned_training_labels.npy').astype(np.uint8)
            print('loaded', self.label.shape[0], 'labels from', path + '/poisoned_training_labels.npy')
        else:
            self.data = (np.concatenate([np.load(x) for x in ls], axis=0) * 255).astype(np.uint8)
            print('loaded', self.data.shape[0],'ULE data from', path)
        self.transform = transform_valid

    def __getitem__(self, index):
        tmp_data = self.data[index]
        img = Image.fromarray(np.transpose(tmp_data, (1, 2, 0)))
        return self.transform(img), self.label[index], index

    def __len__(self):
        return self.data.shape[0]


class TrainDatasetCifar100(Dataset):
    def __init__(self):
        dict = unpickle('data/cifar100/train')
        self.data = dict[b'data'].reshape(50000, 3, 32, 32)
        self.label = dict[b'fine_labels']
        self.transform = transform_valid

    def __getitem__(self, index):
        tmp_data = self.data[index]
        img = Image.fromarray(np.transpose(tmp_data, (1, 2, 0)))
        return self.transform(img), self.label[index], index

    def __len__(self):
        return 50000


class TestDatasetCifar100(Dataset):
    """取每一类前rate比率的测试数据"""

    def __init__(self):
        dict = unpickle('data/cifar100/test')
        self.data = dict[b'data'].reshape(10000, 3, 32, 32)
        self.label = dict[b'fine_labels']
        self.transform = transform_valid

    def __getitem__(self, index):
        tmp_data = self.data[index]
        img = Image.fromarray(np.transpose(tmp_data, (1, 2, 0)))
        return self.transform(img), self.label[index]

    def __len__(self):
        return 10000


def get_data(Train, Test, split_rate, batch_size: int or list, num_workers=4, shuffle=None):
    trainset = Train(split_rate, int(num_workers)) if isinstance(split_rate, str) else Train()#if split_rate > 0 else Train(-split_rate)
    split_rate = 1 if isinstance(split_rate, str) else split_rate #np.abs(split_rate) if split_rate != 0 else 1.0
    testset = Test()
    num_workers = num_workers if isinstance(num_workers, int) else 4
    tmp = int(len(trainset) * split_rate)

    #torch.manual_seed(1000)
    trainset, validset = torch.utils.data.random_split(trainset, [tmp, len(trainset) - tmp])
    trainset.dataset.transform = transform_train

    if shuffle is None:
        shuffles = [True, False, False]
    else:
        shuffles = shuffle

    if isinstance(batch_size, int):
        batch_sizes = [batch_size for _ in range(3)]
    else:
        batch_sizes = batch_size

    sets = [trainset, validset, testset]
    loaders = [DataLoader(sets[i], batch_size=batch_sizes[i], shuffle=shuffles[i], num_workers=num_workers) for i in range(3)]


    return loaders


def cifar10(split_rate, batch_size: int or list, num_workers=4, shuffle=None):
    if isinstance(split_rate, str): 
        return get_data(TrainDatasetCifar10ULE, TestDatasetCifar10, split_rate, batch_size, num_workers=num_workers, shuffle=shuffle)
    return get_data(TrainDatasetCifar10, TestDatasetCifar10, split_rate, batch_size, num_workers=num_workers, shuffle=shuffle)

def cifar100(split_rate, batch_size: int or list, num_workers=4, shuffle=None):
    if isinstance(split_rate, str): 
        return get_data(TrainDatasetCifar100ULE, TestDatasetCifar100, split_rate, batch_size, num_workers=num_workers, shuffle=shuffle)
    return get_data(TrainDatasetCifar100, TestDatasetCifar100, split_rate, batch_size, num_workers=num_workers, shuffle=shuffle)


def get_data_csz(Train, Test, split_rate, batch_size: int or list, num_workers=4, shuffle=None):
    trainset = Train()
    testset = Test()

    if shuffle is None: shuffles = [False, False, False]
    else: shuffles = shuffle

    if isinstance(batch_size, int): batch_sizes = [batch_size for _ in range(3)]
    else: batch_sizes = batch_size

    sets = [trainset, testset]
    loaders = [DataLoader(sets[i], batch_size=batch_sizes[i], shuffle=shuffles[i], num_workers=num_workers) for i in range(2)]
    return loaders[0], None, loaders[1]

def cifar10csz(split_rate, batch_size, num_workers=4, shuffle=None):
    return get_data_csz(TrainDatasetCifar10, TestDatasetCifar10, split_rate, batch_size, num_workers=num_workers, shuffle=shuffle)

def cifar100csz(split_rate, batch_size: int or list, num_workers=4, shuffle=None):
    return get_data_csz(TrainDatasetCifar100, TestDatasetCifar100, split_rate, batch_size, num_workers=num_workers, shuffle=shuffle)