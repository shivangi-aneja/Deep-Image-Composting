"""
    dataset class for all the datasets
"""
import os
import torch
import numpy as np
from torchvision import datasets, transforms
from deep_adversarial_network.utils.custom_dataloader import CustomDataset1,CustomDataset2
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
    elif name == 'toy':
        return  TOY_DATA()
    elif name == 'big':
        return  BIG_DATA()



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

class TOY_DATA(BaseDataset):
    """
    TOY_DATA dataset
    """
    def __init__(self):
        super(BaseDataset, self).__init__()

    def _load(self, dirpath):
        # Training images
        image_tuple = np.load(dirpath + '/toy_data.npy')

        train_data = torch.stack([torch.Tensor(i) for i in image_tuple[0:1000]])
        val_data = torch.stack([torch.Tensor(i) for i in image_tuple[0:5]])

        # train = CustomDataset1(comp_image=train_data[:,0,:,:], fg_img=train_data[:,1,:,:],
        #                       alpha=train_data[:,2,:,:], bg_img=train_data[:,3,:,:])
        # val = CustomDataset1(comp_image=val_data[:, 0, :, :], fg_img=val_data[:, 1, :, :],
        #                       alpha=val_data[:, 2, :, :], bg_img=val_data[:, 3, :, :])

        train = CustomDataset2(comp_image=train_data[:, 0, :, :], gt_img=train_data[:, 1, :, :])
        val = CustomDataset2(comp_image=val_data[:, 0, :, :], gt_img=val_data[:, 1, :, :])
        return train, val

    def n_classes(self):
        # In segmentation n_classes = 2 (One foreground, other background)
        return 2


class BIG_DATA(BaseDataset):
    """
    BIG_DATA dataset
    """
    def __init__(self):
        super(BaseDataset, self).__init__()

    def _load(self, dirpath):
        # Training images
        train_tuple = np.load(dirpath + '/train.npy')
        val_tuple = np.load(dirpath + '/val.npy')

        train_data = torch.stack([torch.Tensor(i) for i in train_tuple[0:2000]])
        val_data = torch.stack([torch.Tensor(i) for i in val_tuple[0:500]])

        train = CustomDataset2(comp_image=train_data[:, 0, :, :], gt_img=train_data[:, 1, :, :])
        val = CustomDataset2(comp_image=val_data[:, 0, :, :], gt_img=val_data[:, 1, :, :])
        return train, val

    def n_classes(self):
        # In segmentation n_classes = 2 (One foreground, other background)
        return 2


DATASETS = {"mnist", "toy", "big"}



