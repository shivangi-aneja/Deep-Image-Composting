"""
    dataset class for all the datasets
"""
from torch.utils.data import Dataset
import PIL.ImageOps
from torchvision import transforms
import numpy as np
import cv2
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def get_available_datasets():
    """
    gets all the datasets
    :return:
    """
    return sorted(DATASETS)


def make_dataset(name, base_path, transform=None):
    """
    it returns dataset according to its name
    :param name: dataset name
    :return: dataset
    """
    name = name.strip().lower()
    if not name in DATASETS:
        raise ValueError("invalid dataset: '{0}'".format(name))
    else:
        return CustomDataset(name=name, base_path=base_path, transform=transform)


def make_datasets(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class CustomDataset(Dataset):
    """
    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    """

    def __init__(self, name, base_path, transform=None, should_invert=False):
        """
            Initialize this dataset class.
        """
        self.comp = os.path.join(base_path + 'comp')  # create a path '/path/to/data/trainA'
        self.gt = os.path.join(base_path + 'gt')  # create a path '/path/to/data/trainB'
        self.comp_paths = sorted(make_datasets(self.comp))  # load images from '/path/to/data/trainA'
        self.gt_paths = sorted(make_datasets(self.gt))  # load images from '/path/to/data/trainB'
        self.comp_size = len(self.comp_paths)  # get the size of composite
        self.gt_size = len(self.gt_paths)  # get the size of ground truth
        self.transform = transform
        self.should_invert = should_invert
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            comp (tensor)     -- composite image
            gt (tensor)       -- ground truth image
        """
        comp_path = self.comp_paths[index % self.comp_size]  # make sure index is within then range
        gt_path = self.gt_paths[index % self.gt_size]

        comp_img = cv2.imread(comp_path)
        comp_img = np.array(comp_img, dtype='uint8')
        comp_img = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB)

        gt_img = cv2.imread(gt_path)
        gt_img = np.array(gt_img, dtype='uint8')
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        if self.should_invert:
            comp_img = PIL.ImageOps.invert(comp_img)
            gt_img = PIL.ImageOps.invert(gt_img)

        if self.transform is not None:
            comp_img_as_tensor = self.transform(comp_img)
            gt_img_as_tensor = self.transform(gt_img)
        else:
            comp_img_as_tensor = self.to_tensor(comp_img)
            gt_img_as_tensor = self.to_tensor(gt_img)

        return comp_img_as_tensor, gt_img_as_tensor

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.comp_size, self.gt_size)


DATASETS = {"coseg"}
