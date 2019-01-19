from __future__ import division

import os

import cv2
import numpy as np
from PIL import Image
from skimage import color
from skimage import io


def change_img_to_png(input_path):
    """
    Changes Image Type to png
    :param input_path: image path
    :return: None
    """
    for image in sorted(os.listdir(input_path)):
        if not image.endswith(".png"):
            os.rename(input_path + "/" + image, input_path + "/" + image.split(".")[0] + '.png')


# Resize images to be of size 400*300
def resize_images(input_path):
    """
    Resize all the images in specific input path
    :param input_path: image path
    :return: None
    """
    new_size = (400, 300)
    for image in sorted(os.listdir(input_path)):
        if image.endswith(".png"):
            img = cv2.imread(input_path + image)
            new_image = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            new_image = cv2.cvtColor(new_image, cv2.IMREAD_COLOR)
            # cv2.waitKey(0)
            cv2.imwrite(input_path + image.split(".")[0] + '.png', new_image)


def save_to_numpy_array(fg_img_path, fg_mask_path, bg_path, path):
    """
    Saves the images in a numpy arrray file
    :param fg_img_path: Foreground Image Path
    :param fg_mask_path: Foreground Mask Path
    :param bg_path: Background Image Path
    :param path: path to store all numpy files
    :return: None
    """
    fg_list = []
    mask_list = []
    bg_list = []

    # For Foreground images
    for image in sorted(os.listdir(fg_img_path)):

        if image.endswith(".png"):
            image = Image.open(fg_img_path + image)
            # This data has shape (height, width, channels)
            data = np.array(image, dtype='uint8')
            # Change to (channels, height, width)
            data = np.transpose(data, [2, 0, 1])
            fg_list.append(data)

    name_fg_img = 'fg_img.npy'
    np.save(os.path.join(path + name_fg_img), fg_list)

    # For Background images
    for image in sorted(os.listdir(bg_path)):

        if image.endswith(".png"):
            image = Image.open(bg_path + image)
            # This data has shape (height, width, channels)
            data = np.array(image, dtype='uint8')
            # Change to (channels, height, width)
            data = np.transpose(data, [2, 0, 1])
            bg_list.append(data)

    name_bg_img = 'bg_img.npy'
    np.save(os.path.join(path + name_bg_img), bg_list)

    # For Foreground Masks
    for image in sorted(os.listdir(fg_mask_path)):

        if image.endswith(".png"):
            # Change the RGB label to Greyscale
            image = color.rgb2gray(io.imread(fg_mask_path + image))
            # This data has shape (height, width)
            data = np.array(image, dtype='uint8')
            mask_list.append(data)
    name_mask = 'fg_mask.npy'
    np.save(os.path.join(path + name_mask), mask_list)


def save_to_numpy(comp_img_path, gt_img_path, path, file):
    """
     Saves the images in a numpy arrray file
    :param comp_img_path: Composite Image Path
    :param gt_img_path:  Ground Truth Image Path
    :param path: path to store numpy file
    :param file: File Name
    :return: None
    """
    img_list = []

    # For Foreground images
    for image in sorted(os.listdir(comp_img_path)):

        if image.endswith(".png"):
            comp_image = Image.open(comp_img_path + image)
            gt_image = Image.open(gt_img_path + image.split(".")[0] + "_blended.png")
            # This data has shape (height, width, channels)
            comp_data = np.array(comp_image, dtype='uint8')
            gt_data = np.array(gt_image, dtype='uint8')
            # Change to (channels, height, width)
            # comp_data = np.transpose(comp_data, [2,0,1])
            # gt_data = np.transpose(gt_data, [2,0,1])
            img_list.append((comp_data, gt_data))

    name_img_file = file + '.npy'
    np.save(os.path.join(path + name_img_file), img_list)


def create_composite_img(comp_img_path, fg_img_path, fg_mask_path, bg_path, comp_file_path):
    """
    Used for old approach (Not required anymore)
    :param comp_img_path:
    :param fg_img_path:
    :param fg_mask_path:
    :param bg_path:
    :param comp_file_path:
    :return:
    """
    composite_img_tuple = []
    # For each foreground
    ctr = 22446
    for fg_img in sorted(os.listdir(fg_img_path))[130:]:
        for bg_img in sorted(os.listdir(bg_path)):

            try:
                if fg_img.endswith(".png") and fg_img is not None and bg_img.endswith(".png") and bg_img is not None:
                    # Load image, mask and background
                    foreground = cv2.imread(fg_img_path + fg_img)
                    alpha = cv2.imread(fg_mask_path + fg_img)
                    background = cv2.imread(bg_path + bg_img)

                    # Convert uint8 to float
                    fg = foreground.astype(float)
                    bg = background.astype(float)

                    # Normalize the alpha mask to keep intensity between 0 and 1
                    alpha = alpha.astype(float) / 255

                    # Multiply the foreground with the alpha matte
                    foreground = cv2.multiply(alpha, fg)

                    # Multiply the background with ( 1 - alpha )
                    background = cv2.multiply(1.0 - alpha, bg)

                    # Add the masked foreground and background.
                    out_image = cv2.add(foreground, background)

                    # Display image
                    cv2.imwrite(comp_img_path + str(ctr) + '.png', out_image)
                    cv2.imwrite(comp_img_path + str(ctr) + '_fg.png', fg)
                    cv2.imwrite(comp_img_path + str(ctr) + '_bg.png', bg)
                    cv2.imwrite(comp_img_path + str(ctr) + '_mask.png', alpha * 255)
                    # composite_img_tuple.append((out_image, fg, alpha*255, bg))
                    ctr += 1
                    print(ctr)
            except:
                print(fg_img_path + fg_img)
                print(fg_mask_path + fg_img)
                print(bg_path + bg_img)

    # name_mask = 'composite.npy'
    # np.save(os.path.join(comp_file_path + name_mask), composite_img_tuple)


def create_composite_img_new(comp_img_path, mask_path, gt_img_path, data_path, comp_file_path, file_name):
    """
    Create Composite Image and saves them in as tuple (Composite Image, Ground Truth) in npy file
    :param comp_img_path: Composite Image Path
    :param mask_path: Foreground Mask Path
    :param gt_img_path: Ground Truth Image Path
    :param data_path: Path to write composite and ground truth image pair
    :param comp_file_path: Composite Image File Path
    :param file_name: Composite Image File Name
    :return: None
    """
    composite_img_tuple = []
    ctr = 0
    for fg_img in sorted(os.listdir(comp_img_path)):

        try:
            if fg_img.endswith(".png") and fg_img is not None:
                # Load image, mask and background
                foreground = cv2.imread(comp_img_path + fg_img)
                alpha = cv2.imread(mask_path + fg_img.split("_style")[0] + ".png")
                background = cv2.imread(gt_img_path + fg_img.split("_style")[0] + ".png")

                # Convert uint8 to float
                fg = foreground.astype(float)
                bg = background.astype(float)

                # Normalize the alpha mask to keep intensity between 0 and 1
                alpha = alpha.astype(float) / 255

                # Multiply the foreground with the alpha matte
                foreground = cv2.multiply(alpha, fg)

                # Multiply the background with ( 1 - alpha )
                background = cv2.multiply(1.0 - alpha, bg)

                # Add the masked foreground and background.
                out_image = cv2.add(foreground, background)

                # Display image
                # BGR to RGB while writing to numpy file
                cv2.imwrite(data_path + str(ctr) + '_cp.png', out_image)
                cv2.imwrite(data_path + str(ctr) + '_gt.png', bg)
                composite_img_tuple.append((out_image[..., [2, 1, 0]], bg[..., [2, 1, 0]]))
                ctr += 1
                print(ctr)
        except:
            print(comp_img_path + fg_img)
            print(mask_path + fg_img)

    np.save(os.path.join(comp_file_path + file_name), composite_img_tuple)


def main():
    """
    Main function to create composite images
    :return: None
    """
    gt_dir = os.path.join(os.getcwd(), 'data/big_data/ground_truth/')
    mask_dir = os.path.join(os.getcwd(), 'data/big_data/mask/')

    train_dir_data = os.path.join(os.getcwd(), 'data/big_data/train/data/')
    train_dir_comp = os.path.join(os.getcwd(), 'data/big_data/train/composite/')

    val_dir_data = os.path.join(os.getcwd(), 'data/big_data/val/data/')
    val_dir_comp = os.path.join(os.getcwd(), 'data/big_data/val/composite/')

    path = os.path.join(os.getcwd(), 'data/big_data/')

    # For renaming
    # change_img_to_png(input_dir4)
    # change_img_to_png(input_dir2)
    # change_img_to_png(input_dir3)
    #
    # # For resizing
    # resize_images(input_path = gt_dir)
    # resize_images(input_path = mask_dir)
    # resize_images(input_path = train_dir_comp)
    # resize_images(input_path = val_dir_comp)

    # save_to_numpy(comp_img_path=input_dir4, gt_img_path=input_dir5,  path=path, file='big_data')
    # Save to numpy
    # save_to_numpy_array(fg_img_path=input_dir3, fg_mask_path=input_dir2, bg_path=input_dir1, path=path)

    # Create Composite Images Train
    create_composite_img_new(comp_img_path=train_dir_comp, gt_img_path=gt_dir, mask_path=mask_dir,
                             data_path=train_dir_data,
                             comp_file_path=path, file_name="train.npy")

    # Create Composite Images Val
    create_composite_img_new(comp_img_path=val_dir_comp, gt_img_path=gt_dir, mask_path=mask_dir,
                             data_path=val_dir_data,
                             comp_file_path=path, file_name="val.npy")


if __name__ == '__main__':
    main()
