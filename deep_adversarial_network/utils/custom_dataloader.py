from torch.utils.data import Dataset
import torch

class CustomDataset1(Dataset):
    def __init__(self, comp_image, fg_img, alpha, bg_img):
        self.comp_image = comp_image
        self.fg_img = fg_img
        self.alpha = alpha
        self.bg_img = bg_img


    def __getitem__(self, index):
        comp_image = self.comp_image[index]
        fg_img = self.fg_img[index]
        alpha = self.alpha[index]
        bg_img = self.bg_img[index]

        return comp_image, fg_img, alpha, bg_img

    def __len__(self):
        return len(self.comp_image)

class CustomDataset2(Dataset):
    def __init__(self, comp_image, gt_img):
        self.comp_image = comp_image
        self.gt_img = gt_img


    def __getitem__(self, index):
        comp_image = self.comp_image[index]
        gt_img = self.gt_img[index]


        return comp_image, gt_img

    def __len__(self):
        return len(self.comp_image)