#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Main file to train and evaluate the models
"""

from deep_adversarial_network.metrics.metric_eval import *
from PIL import Image
import numpy as np
import os
from skimage.measure import compare_ssim as ssim


def main():
    """
    main function that parses the arguments and trains
    :param args: arguments related
    :return: None
    """
    mse_avg_total = 0.
    psnr_avg_total = 0.
    ssim_avg_total = 0.
    vif_avg_total = 0.

    image_list = ['4','36','39','70','121','149']
    for img in image_list:

        gt_image =  Image.open(os.getcwd()+'/metrics/gt/gt_'+img+'.png')
        gt_image = np.array(gt_image)
        test_image = Image.open(os.getcwd() + '/metrics/ours/ht_' + img + '.png')
        test_image = np.array(test_image)

        mse_avg_iter, psnr_avg_iter = calc_mse_psnr_img(test_image, gt_image)
        #tv_avg_iter = get_total_variation(tf.convert_to_tensor(test_image))
        ssim_avg_iter = ssim(test_image,gt_image,multichannel=True)
        vif_avg_iter = calc_vif_img(test_image, gt_image)

        mse_avg_total += mse_avg_iter
        psnr_avg_total += psnr_avg_iter
        #tv_avg_total += tv_avg_iter
        ssim_avg_total += ssim_avg_iter
        vif_avg_total += vif_avg_iter

    mse_avg_total /= len(image_list)
    psnr_avg_total /= len(image_list)
    ssim_avg_total /= len(image_list)
    vif_avg_total /= len(image_list)

    print("MSE : %.3f, PSNR : %.3f, SSIM : %.3f  VIF: %3F" % (
        mse_avg_total, psnr_avg_total, ssim_avg_total, vif_avg_total))


if __name__ == '__main__':
    main()
