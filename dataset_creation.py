import numpy as np
import os
import cv2


def generate_masks():
    """
    :return:
    """
    base_path = 'voc/'

    for img_path in sorted(os.listdir(base_path)):
        img = cv2.imread(base_path + '/' + img_path, 2)
        ret, bw_img = cv2.threshold(src=img, thresh=0, maxval=255, type=cv2.THRESH_BINARY)
        cv2.imwrite('voc_masks/' + img_path.split(".")[0] + '_cp.png', bw_img)


def copy_images_of_mask():
    """
    :return:
    """
    base_path = 'voc/masks/'
    jpg_path = 'voc/JPEGImages/'
    gt_path = 'voc/gt/'

    for img_path in sorted(os.listdir(base_path)):
        img_name = img_path.split("_cp")[0]
        img = cv2.imread(jpg_path + '/' + img_name + '.jpg')
        cv2.imwrite(gt_path + '/' + img_name + '.png', img)


def create_msrc_mask():
    base_path = 'msrc/masks/'
    mask_path = 'msrc/mask_new/'
    for img_path in sorted(os.listdir(base_path)):
        img = cv2.imread(base_path + '/' + img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # green (grass) to black
        # lower = (0, 128, 0)  # lower bound for each channel
        # upper = (0, 128, 0)  # upper bound for each channel
        # mask = cv2.inRange(img, lower, upper)

        # cow (blue) to white
        # lower = (0, 0, 128)  # lower bound for each channel
        # upper = (0, 0, 128)  # upper bound for each channel
        # mask = cv2.inRange(img, lower, upper)

        # sheep (sky blue) to white
        # lower = (0, 128, 128)  # lower bound for each channel
        # upper = (0, 128, 128)  # upper bound for each channel
        # mask = cv2.inRange(img, lower, upper)

        # water to black
        # lower = (64, 128, 0)  # lower bound for each channel
        # upper = (64, 128, 0)  # upper bound for each channel
        # mask = cv2.inRange(img, lower, upper)

        # sky to black
        # lower = (128, 128, 128)  # lower bound for each channel
        # upper = (128, 128, 128)  # upper bound for each channel
        # mask = cv2.inRange(img, lower, upper)

        # building to white
        # lower = (128, 0, 0)  # lower bound for each channel
        # upper = (128, 0, 0)  # upper bound for each channel
        # mask = cv2.inRange(img, lower, upper)

        # horse to white
        # lower = (128, 0, 128)  # lower bound for each channel
        # upper = (128, 0, 128)  # upper bound for each channel
        # mask = cv2.inRange(img, lower, upper)

        # aeroplane to white
        # lower = (192, 0, 0)  # lower bound for each channel
        # upper = (192, 0, 0)  # upper bound for each channel
        # mask = cv2.inRange(img, lower, upper)

        # car to white
        # lower = (64, 0, 128)  # lower bound for each channel
        # upper = (64, 0, 128)  # upper bound for each channel
        # mask = cv2.inRange(img, lower, upper)

        # animal to white
        # lower = (64, 128, 64)  # lower bound for each channel
        # upper = (64, 128, 64)  # upper bound for each channel
        # mask = cv2.inRange(img, lower, upper)

        #
        lower = (64, 128, 64)  # lower bound for each channel
        upper = (64, 128, 64)  # upper bound for each channel
        mask = cv2.inRange(img, lower, upper)

        img[mask != 0] = [0, 0, 0]  # Black
        # img[mask != 0] = [255, 255, 255]  # White
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_path + '/' + img_path.split(".")[0] + '.png', img)

def count_unique_colors():
    base_path = 'msrc/gt/'

    for img_path in sorted(os.listdir(base_path)):
        # img_path = '8_28_s_GT.png'
        img = cv2.imread(base_path + '/' + img_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #unique, counts = np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
        # print(unique)
        # print(counts)

        # lower = (255, 255, 255)  # lower bound for each channel
        # upper = (255, 255, 255)  # upper bound for each channel
        # mask = cv2.inRange(img, lower, upper)
        # img[mask != 0] = [0, 0, 0]  # Black

        # green
        # lower = (0, 128, 0)  # lower bound for each channel
        # upper = (0, 128, 0)  # upper bound for each channel
        # mask = cv2.inRange(img, lower, upper)
        # img[mask != 0] = [0, 0, 0]  # White

        # flower
        # lower = (64, 128, 128)  # lower bound for each channel
        # upper = (64, 128, 128)  # upper bound for each channel
        # mask = cv2.inRange(img, lower, upper)
        # img[mask != 0] = [255, 255, 255]  # White

        # purple
        # lower = (128, 64, 128)  # lower bound for each channel
        # upper = (128, 64, 128)  # upper bound for each channel
        # mask = cv2.inRange(img, lower, upper)
        # img[mask != 0] = [0, 0, 0]  # Black
        #
        # # tree
        # lower = (128, 128, 0)  # lower bound for each channel
        # upper = (128, 128, 0)  # upper bound for each channel
        # mask = cv2.inRange(img, lower, upper)
        # img[mask != 0] = [255, 255, 255]  # Black
        #
        # human face
        # lower = (64, 64, 0)  # lower bound for each channel
        # upper = (64, 64, 0)  # upper bound for each channel
        # mask = cv2.inRange(img, lower, upper)
        # img[mask != 0] = [255, 255, 255]  # White

        # human body
        # lower = (192, 128, 0)  # lower bound for each channel
        # upper = (192, 128, 0)  # upper bound for each channel
        # mask = cv2.inRange(img, lower, upper)
        # img[mask != 0] = [255, 255, 255]  # White

        # background
        # lower = (128, 64, 0)  # lower bound for each channel
        # upper = (128, 64, 0)  # upper bound for each channel
        # mask = cv2.inRange(img, lower, upper)
        # img[mask != 0] = [0, 0, 0]  # Black



        # lower = (128, 128, 128)  # lower bound for each channel
        # upper = (128, 128, 128)  # upper bound for each channel
        # mask = cv2.inRange(img, lower, upper)
        # img[mask != 0] = [0, 0, 0]
        #
        # #
        # lower = (128, 128, 0)  # lower bound for each channel
        # upper = (128, 128, 0)  # upper bound for each channel
        # mask = cv2.inRange(img, lower, upper)
        # img[mask != 0] = [0, 0, 0]

        # # img[mask != 0] = [255, 255, 255]  # White
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(base_path + '/' + img_path.split(".")[0] + '.png', img)


if __name__ == '__main__':
    # create_msrc_mask()
    count_unique_colors()
