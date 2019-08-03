#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Main file to train and evaluate the models
"""

import argparse
import os
import tensorflow as tf
from torch.utils.data import DataLoader
from torchvision import transforms
from deep_adversarial_network.adversarial_training import DeepGAN
from deep_adversarial_network.discriminator import (get_available_discriminators, make_discriminator)
from deep_adversarial_network.generator import (get_available_generators, make_generator)
from deep_adversarial_network.logging.logger import rootLogger
from deep_adversarial_network.utils import (get_available_datasets,
                                            make_dataset)


# Optimizers
OPTIMIZERS = {
    'adam': tf.train.AdamOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'sgd': tf.train.GradientDescentOptimizer,
    'rms_prop': tf.train.RMSPropOptimizer
}

# Losses
LOSSES = {
    'l1': tf.losses.absolute_difference,
    'l2': tf.losses.mean_squared_error
}

# General Paths
LOG_PATH = os.path.join(os.getcwd(), 'logs/')
PLOT_PATH = os.path.join(os.getcwd(), 'plots/')
MODEL_PATH = os.path.join(os.getcwd(), 'models/')

# training settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# general
parser.add_argument('-d', '--dataset', type=str, default='coseg',
                    help="dataset, {'" + \
                         "', '".join(get_available_datasets()) + \
                         "'}")
parser.add_argument('--data-dirpath', type=str, default='data/',
                    help='directory for storing downloaded data')

parser.add_argument('--n-workers', type=int, default=2,
                    help='how many threads to use for I/O')

parser.add_argument('--gpu', type=str, default='0',
                    help="ID of the GPU to train on (or '' to train on CPU)")

parser.add_argument('-rs', '--random-seed', type=int, default=1,
                    help="random seed for training")

# GAN-related
parser.add_argument('-dr', '--discriminator', type=str, default='patch',
                    help="discriminator architecture name, {'" + \
                         "', '".join(get_available_discriminators()) + \
                         "'}")

parser.add_argument('-gr', '--generator', type=str, default='multi2',
                    help="generator architecture name, {'" + \
                         "', '".join(get_available_generators()) + \
                         "'}")

parser.add_argument('-d_lr', '--d_lr', type=float, default=1e-4,
                    help='discriminator learning rate')

parser.add_argument('-g_lr', '--g_lr', type=float, default=1e-4,
                    help='generator learning rate')

parser.add_argument('-b', '--batch_size', type=int, default=16,
                    help='input batch size for training')

parser.add_argument('-d_opt', '--d_optim', type=str, default='adam',
                    help="optimizer, {'" + \
                         "', '".join(OPTIMIZERS.keys()) + \
                         "'}")

parser.add_argument('-g_opt', '--g_optim', type=str, default='adam',
                    help="optimizer, {'" + \
                         "', '".join(OPTIMIZERS.keys()) + \
                         "'}")
parser.add_argument('-m', '--model_name', type=str,
                    default='gan_model', help='name for model')

parser.add_argument('-e', '--epochs', type=int, default=1000,
                    help='number of epochs')

parser.add_argument('-rl', '--recon_loss', type=str, default='l2',
                    help="losses, {'" + \
                         "', '".join(LOSSES.keys()) + \
                         "'}")

# Plot related
parser.add_argument('-tf', '--tf_logs', type=str, default='tf_logs',
                    help="log folder for tensorflow logging")

parser.add_argument('-mp', '--plot_matplotlib', type=str, default='n',
                    help="whether to plot matplotlib plots")

# parse and validate parameters
args = parser.parse_args()

for k, v in args._get_kwargs():
    if isinstance(v, str):
        setattr(args, k, v.strip().lower())

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# print arguments
rootLogger.info("Running with the following parameters:")
rootLogger.info(vars(args))


def main(args=args):
    """
    main function that parses the arguments and trains
    :param args: arguments related
    :return: None
    """
    batch_size = args.batch_size
    mplib = True if args.plot_matplotlib == 'y' else False

    # load and shuffle data
    train_dataset = make_dataset(name=args.dataset, base_path=os.getcwd() + '/data/' + args.dataset + '/train/',
                                 transform=transforms.Compose(
                                     [transforms.ToPILImage(), transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]))
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=args.n_workers)
    val_dataset = make_dataset(name=args.dataset, base_path=os.getcwd() + '/data/' + args.dataset + '/val/',
                               transform=transforms.Compose(
                                   [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]))
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=args.n_workers)

    recon_loss = LOSSES.get(args.recon_loss, None)
    if not recon_loss:
        raise ValueError("invalid loss: '{0}'".format(args.recon_loss))

    tf_log_path = os.path.join(os.getcwd(), args.tf_logs + '/')

    # build discriminator model
    discriminator = make_discriminator(name=args.discriminator)

    # build generator model
    generator = make_generator(name=args.generator)

    # get optimizer
    d_optim = OPTIMIZERS.get(args.d_optim, None)
    if not d_optim:
        raise ValueError("invalid optimizer: '{0}'".format(args.d_optim))

    g_optim = OPTIMIZERS.get(args.g_optim, None)
    if not g_optim:
        raise ValueError("invalid optimizer: '{0}'".format(args.g_optim))

    # get learning rate
    d_lr = args.d_lr
    g_lr = args.g_lr

    # Create GAN according to params
    model = DeepGAN(discriminator=discriminator, generator=generator, model_name=args.model_name, recon_loss=recon_loss,
                    dataset=args.dataset, batch_size=args.batch_size, d_optim=d_optim, g_optim=g_optim, d_lr=d_lr,
                    g_lr=g_lr,
                    epochs=args.epochs, mplib=mplib, tf_log_path=tf_log_path)
    # Train the model
    model.adversarial_train(train_loader=train_loader, test_loader=val_loader, model_path=MODEL_PATH)


if __name__ == '__main__':
    main()
