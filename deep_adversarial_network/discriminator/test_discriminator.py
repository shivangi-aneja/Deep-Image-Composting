"""
discriminator for mnist dataset
"""
from deep_adversarial_network.utils.common_util import *


class test_Discriminator1(object):
    """
    test_Discriminator1
    """
    def __init__(self):
        pass

    def make_discriminator_network(self, x, reuse=False,isTrain=True):
        with tf.variable_scope("discriminator", reuse=reuse):
            #input = tf.placeholder(tf.float32, (None, 16, 16, 3), name="input")
            conv1 = tf.layers.conv2d(inputs=x, filters=16, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
            conv1_bn = tf.layers.batch_normalization(conv1)

            conv2 = tf.layers.conv2d(inputs=conv1_bn, filters=32, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
            conv2_bn = tf.layers.batch_normalization(conv2)

            conv3 = tf.layers.conv2d(inputs=conv2_bn, filters=64, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
            conv3_bn = tf.layers.batch_normalization(conv3)

            fc4_reshape = tf.reshape(conv3_bn, shape=[-1, 16 * 16 * 64])
            fc4 = tf.layers.dense(fc4_reshape, units=1)
            out = tf.nn.sigmoid(fc4)


        return fc4, out

