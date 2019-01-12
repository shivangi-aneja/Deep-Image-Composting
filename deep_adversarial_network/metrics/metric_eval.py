import numpy as np
import math

PIXEL_MAX = 255.0

def calc_mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def calc_mse_psnr(img_list1, img_list2):

    num_imgs = img_list1.shape[0]

    total_mse = 0.
    total_psnr = 0.

    for i in range(num_imgs):
        mse_val = calc_mse(img_list1[i],img_list2[i])
        if mse_val == 0:
            psnr = 100
        else:
            psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse_val))

        total_mse += mse_val
        total_psnr += psnr

    return  total_mse/num_imgs, total_psnr/num_imgs