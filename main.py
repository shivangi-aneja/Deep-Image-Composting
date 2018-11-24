#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Main file to train and evaluate the models
"""

import os
import argparse
import pprint
import tensorflow as tf
from torch.utils.data import DataLoader
from deep_adversarial_network.logging.logger import rootLogger
from deep_adversarial_network.utils import (get_available_datasets,
                                   make_dataset, RNG)
from deep_adversarial_network.discriminator import (get_available_discriminators,make_discriminator)
from deep_adversarial_network.generator import (get_available_generators,make_generator)
from deep_adversarial_network.utils.pytorch_dataset_utils import DatasetIndexer
from deep_adversarial_network.adversarial_training import DeepGAN

# Optimizers
OPTIMIZERS = {
    'adam': tf.train.AdamOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'sgd': tf.train.GradientDescentOptimizer,
    'rms_prop': tf.train.RMSPropOptimizer
}

LOG_PATH = os.path.join(os.getcwd(), 'logs/')
PLOT_PATH = os.path.join(os.getcwd(), 'plots/')

NOISE_SIZE = 100
IMAGE_PIXELS = 28*28

# training settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# general
parser.add_argument('-d', '--dataset', type=str, default='toy',
                    help="dataset, {'" +\
                         "', '".join(get_available_datasets()) +\
                         "'}")
parser.add_argument('--data-dirpath', type=str, default='data/',
                    help='directory for storing downloaded data')

parser.add_argument('--n-workers', type=int, default=2,
                    help='how many threads to use for I/O')

parser.add_argument('--gpu', type=str, default='0',
                    help="ID of the GPU to train on (or '' to train on CPU)")

parser.add_argument('-rs', '--random-seed', type=int, default=1,
                    help="random seed for training")

parser.add_argument('-tf', '--tf_logs', type=str, default='tf_logs',
                    help="log folder for tensorflow logging")

# GAN-related
parser.add_argument('-dr', '--discriminator', type=str, default='mnist_discriminator1',
                    help="discriminator architecture name, {'" + \
                         "', '".join(get_available_discriminators()) +\
                         "'}")

parser.add_argument('-gr', '--generator', type=str, default='mnist_generator1',
                    help="generator architecture name, {'" + \
                         "', '".join(get_available_generators()) +\
                         "'}")

parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4,
                    help='initial learning rate')

parser.add_argument('-b', '--batch_size', type=int, default=256,
                    help='input batch size for training')

parser.add_argument('-opt', '--optim', type=str, default='adam',
                    help="optimizer, {'" + \
                         "', '".join(OPTIMIZERS.keys()) +\
                         "'}")
parser.add_argument('-m', '--model_name', type=str,
                    default='gan_model', help='name for model')

parser.add_argument('-e', '--epochs', type=int, default=5,
                    help='number of epochs')




# parse and validate parameters
args = parser.parse_args()

for k, v in args._get_kwargs():
    if isinstance(v, str):
        setattr(args, k, v.strip().lower())

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# print arguments
rootLogger.info("Running with the following parameters:")
pprint.pprint(vars(args))

def main(args=args):
    """
    main function that parses the arguments and trains
    :param args: arguments related
    :return: None
    """

    # load and shuffle data
    dataset = make_dataset(args.dataset)

    train_dataset, val_dataset = dataset.load(args.data_dirpath)

    rng = RNG(args.random_seed)
    train_ind = rng.permutation(len(train_dataset))
    val_ind = rng.permutation(len(val_dataset))

    train_dataset = DatasetIndexer(train_dataset, train_ind)
    val_dataset = DatasetIndexer(val_dataset, val_ind)

    batch_size = args.batch_size

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=args.n_workers)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=args.n_workers)

    print("Main function")
    tf_log_path = os.path.join(os.getcwd(), args.tf_logs+'/')
    # Input
    X = tf.placeholder(tf.float32, shape=(None, IMAGE_PIXELS))
    # Input
    Z = tf.placeholder(tf.float32, shape=(None, NOISE_SIZE))

    # build discriminator model
    discriminator = make_discriminator(name=args.discriminator, x=X)

    # build generator model
    generator = make_generator(name=args.generator, z=Z)

    # get optimizer
    optim = OPTIMIZERS.get(args.optim, None)
    if not optim:
        raise ValueError("invalid optimizer: '{0}'".format(args.optim))


    # Create GAN according to params
    model = DeepGAN(discriminator=discriminator, generator=generator, noise_size=NOISE_SIZE, model_name=args.model_name,
                    dataset=args.dataset, batch_size=args.batch_size, optim=optim, lr=args.learning_rate, epochs=args.epochs)
    # Train the model
    model.adversarial_train(X=X, Z=Z, data_loader=train_loader, tf_log_path=tf_log_path)


if __name__ == '__main__':
    main()