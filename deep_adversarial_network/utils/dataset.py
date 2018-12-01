"""
    dataset class for all the datasets
"""
import os
import torch
import numpy as np
from torchvision import datasets, transforms
from deep_adversarial_network.utils.custom_dataloader import CustomDataset
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
    elif name == 'toy':
        return  TOY_DATA()


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

class TOY_DATA(BaseDataset):
    """
    TOY_DATA dataset
    """
    def __init__(self):
        super(BaseDataset, self).__init__()

    def _load(self, dirpath):
        # Training images
        image_tuple = np.load(dirpath + '/composite.npy')

        train_data = torch.stack([torch.Tensor(i) for i in image_tuple[0:10]])
        val_data = torch.stack([torch.Tensor(i) for i in image_tuple[10:20]])

        train = CustomDataset(comp_image=train_data[:,0,:,:], fg_img=train_data[:,1,:,:],
                              alpha=train_data[:,2,:,:], bg_img=train_data[:,3,:,:])
        val = CustomDataset(comp_image=val_data[:, 0, :, :], fg_img=val_data[:, 1, :, :],
                              alpha=val_data[:, 2, :, :], bg_img=val_data[:, 3, :, :])
        return train, val

    def n_classes(self):
        # In segmentation n_classes = 2 (One foreground, other background)
        return 2


DATASETS = {"mnist", "toy"}



