"""
    dataset class for all the datasets
"""
import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.dataset import (Subset)

def get_available_datasets():
    """
    gets all the datasets
    :return:
    """
    return sorted(DATASETS)


def make_dataset(name):
    """
    it returns dataset according to its name
    :param name: dataset name
    :return: dataset
    """
    name = name.strip().lower()
    if not name in DATASETS:
        raise ValueError("invalid dataset: '{0}'".format(name))
    elif name == 'mnist':
        return MNIST()


class BaseDataset(object):
    """
    base dataset
    """
    def _load(self, dirpath):
        """Download if needed and return the dataset.
        Return format is (`torch.Tensor` data, `torch.Tensor` label) iterable
        of appropriate shape or tuple (train data, test data) of such.
        """
        raise NotImplementedError('`load` is not implemented')

    def load(self, dirpath):
        """
        loads the dataset
        :param dirpath: directory path where dataset is stored
        :return:
        """
        return self._load(os.path.join(dirpath, self.__class__.__name__.lower()))

    def n_classes(self):
        """Get number of classes."""
        raise NotImplementedError('`n_classes` is not implemented')

class MNIST(BaseDataset):
    """
    MNIST dataset
    """
    def __init__(self):
        super(BaseDataset, self).__init__()


    def _load(self, dirpath):
        # Normalized images
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_mask = range(55000)
        val_mask = range(55000, 60000)
        train_val = datasets.MNIST(root=dirpath, train=True, download=True, transform=trans)
        train = Subset(train_val, train_mask)
        val = Subset(train_val, val_mask)
        test = datasets.MNIST(root=dirpath, train=False, download=True, transform=trans)
        return train, val, test

    def n_classes(self):
        return 10

class POLYPS(BaseDataset):
    """
    POLYPS dataset
    """
    def __init__(self):
        super(BaseDataset, self).__init__()

    def _load(self, dirpath):
        # Training images
        train_img = np.load(dirpath + '/train_X.npy')
        train_val_y = np.load(dirpath + '/train_Y.npy')

        train_x = torch.stack([torch.Tensor(i) for i in train_val_x[0:800]])
        val_x = torch.stack([torch.Tensor(i) for i in train_val_x[800:]])

        train_y = torch.stack([torch.Tensor(i) for i in train_val_y[0:800]])
        val_y = torch.stack([torch.Tensor(i) for i in train_val_y[800:]])

        test_x = torch.stack([torch.Tensor(i) for i in np.load(dirpath + '/test_X.npy')])
        test_y = torch.stack([torch.Tensor(i) for i in np.load(dirpath + '/test_Y.npy')])

        train = SegDataset(train_x, train_y)
        val = SegDataset(val_x, val_y)
        test = SegDataset(test_x,test_y)

        return train, val, test

    def n_classes(self):
        # In segmentation n_classes = 2 (One foreground, other background)
        return 2


DATASETS = {"polyps"}


DATASETS = {"mnist"}



