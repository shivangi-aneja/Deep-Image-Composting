"""
discriminator for mnist dataset
"""
from deep_adversarial_network.utils.common_util import *


class test_Discriminator1(object):
    """
    Big Image
    """
    def __init__(self):
        pass

    def make_discriminator_network(self, x, reuse=False,isTrain=True):
        with tf.variable_scope("discriminator", reuse=reuse):
            #input = tf.placeholder(tf.float32, (None, 16, 16, 3), name="input")
            conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv1_bn = tf.layers.batch_normalization(conv1)

            conv2 = tf.layers.conv2d(inputs=conv1_bn, filters=64, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv2_bn = tf.layers.batch_normalization(conv2)

            # conv3 = tf.layers.conv2d(inputs=conv2_bn, filters=128, kernel_size=(3, 3), padding='same',
            #                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            # conv3_bn = tf.layers.batch_normalization(conv3)
            #
            # conv4 = tf.layers.conv2d(inputs=conv3_bn, filters=256, kernel_size=(3, 3), padding='same',
            #                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            # conv4_bn = tf.layers.batch_normalization(conv4)

            fc4_reshape = tf.reshape(conv2_bn, shape=[-1, 300 * 400 * 64])
            logits = tf.layers.dense(fc4_reshape, units=1)
            out = tf.nn.sigmoid(logits)

        return out, logits

class test_Discriminator2(object):
    """
    working
    """
    def __init__(self):
        pass

    def make_discriminator_network(self, x, reuse=False,isTrain=True):
        with tf.variable_scope("discriminator", reuse=reuse):
            #input = tf.placeholder(tf.float32, (None, 16, 16, 3), name="input")
            conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv1_bn = tf.layers.batch_normalization(conv1)

            conv2 = tf.layers.conv2d(inputs=conv1_bn, filters=64, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv2_bn = tf.layers.batch_normalization(conv2)

            conv3 = tf.layers.conv2d(inputs=conv2_bn, filters=128, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv3_bn = tf.layers.batch_normalization(conv3)

            conv4 = tf.layers.conv2d(inputs=conv3_bn, filters=256, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv4_bn = tf.layers.batch_normalization(conv4)

            fc4_reshape = tf.reshape(conv4_bn, shape=[-1, 32 * 32 * 256])
            logits = tf.layers.dense(fc4_reshape, units=1)
            out = tf.nn.sigmoid(logits)

        return out, logits


class Resnet_Discriminator(object):
    """
    test_Discriminator1
    """
    def __init__(self):
        pass

    def make_discriminator_network(self, x, reuse=False, isTrain=True):
        with tf.variable_scope("discriminator", reuse=reuse):
            #input = tf.placeholder(tf.float32, (None, 16, 16, 3), name="input")
            conv1 = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(7, 7), padding='valid', strides=2,
                                     activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv1_maxpool = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            #Block1

            resnet_conv1 = tf.layers.conv2d(inputs=conv1_maxpool, filters=64, kernel_size=(3, 3), padding='valid', strides=2,
                                     activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

            resnet_conv1_bn = tf.layers.batch_normalization(resnet_conv1)
            resnet_conv1_bn = tf.nn.relu(resnet_conv1_bn)

            resnet_conv2 = tf.layers.conv2d(inputs=resnet_conv1_bn, filters=64, kernel_size=(3, 3), padding='valid',
                                            strides=2,activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

            resnet_conv2_bn = tf.layers.batch_normalization(resnet_conv2)
            resnet_conv2_bn = tf.nn.relu(resnet_conv2_bn)

            resnet_conv2_bn += conv1_maxpool
            resnet_conv2_bn = tf.nn.relu(resnet_conv2_bn)

            #Block2

            resnet2_conv1 = tf.layers.conv2d(inputs=resnet_conv2_bn, filters=64, kernel_size=(3, 3), padding='valid',
                                            strides=2, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

            resnet2_conv1_bn = tf.layers.batch_normalization(resnet2_conv1)
            resnet2_conv1_bn = tf.nn.relu(resnet2_conv1_bn)

            resnet2_conv2 = tf.layers.conv2d(inputs=resnet2_conv1_bn, filters=64, kernel_size=(3, 3), padding='valid',
                                            strides=2, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

            resnet2_conv2_bn = tf.layers.batch_normalization(resnet2_conv2)
            resnet2_conv2_bn = tf.nn.relu(resnet2_conv2_bn)

            resnet2_conv2_bn += resnet_conv2_bn
            resnet2_conv2_bn = tf.nn.relu(resnet2_conv2_bn)


            reshape = tf.reshape(resnet2_conv2_bn, shape=[-1, 2048])

            logits = tf.layers.dense(reshape, units=1)
            out = tf.nn.sigmoid(logits)

        return out, logits

