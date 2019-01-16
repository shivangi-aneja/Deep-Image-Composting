import numpy as np
import math
import tensorflow as tf

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
        im1 = img_list1[i]
        im2 = img_list2[i].numpy()
        mse_val = calc_mse(im1,im2)
        if mse_val == 0.:
            psnr = 100
        else:
            # im1 = tf.image.convert_image_dtype(img_list1[i], tf.float32)
            # im2 = tf.image.convert_image_dtype(img_list2[i], tf.float32)
            # psnr = tf.image.psnr(im1, im2, max_val=255)
            psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse_val))

        total_mse += mse_val
        total_psnr += psnr

    return  total_mse/num_imgs, total_psnr/num_imgs


def d_accuracy(real_prob, fake_prob):
    label_real = np.ones(shape = real_prob.shape[0])
    label_fake = np.zeros(shape = fake_prob.shape[0])
    real_pred = np.round(real_prob)
    fake_pred = np.round(fake_prob)
    acc_real = (label_real == real_pred).mean()
    acc_fake = (label_fake == fake_pred).mean()
    return np.mean([acc_real, acc_fake])


