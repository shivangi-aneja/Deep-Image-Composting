"""
generator network
"""

from deep_adversarial_network.utils.common_util import *
from tensorflow.python.ops import random_ops

class test_Generator1():
    """
    MNIST_Discriminator1
    """
    def __init__(self):
        pass

    def make_generator_network(self, mask, reuse=False, isTrain=True):
        with tf.variable_scope("generator", reuse=reuse):
            input = mask
            # inputs_ = tf.placeholder(tf.float32, (None, 16, 16, 3), name="input")
            # targets_ = tf.placeholder(tf.float32, (None, 16, 16, 1), name="target")

            ### Encoder
            conv1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.leaky_relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv1_bn = tf.layers.batch_normalization(conv1)

            conv2 = tf.layers.conv2d(inputs=conv1_bn, filters=128, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv2_bn = tf.layers.batch_normalization(conv2)

            conv3 = tf.layers.conv2d(inputs=conv2_bn, filters=256, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv3_bn = tf.layers.batch_normalization(conv3)

            ### Bottleneck

            fc4_reshape = tf.reshape(conv3_bn, shape = [ -1,32*32*256])
            fc4 = tf.layers.dense(fc4_reshape, units=1024)

            fc5 = tf.layers.dense(fc4, units=32*32*256)

            fc5_reshape = tf.reshape(fc5, shape=[-1, 32 , 32 , 256])


            ### Decoder

            deconv3 = tf.layers.conv2d_transpose(inputs=fc5_reshape, filters=256, kernel_size=(3, 3), padding='same',
                                                 activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            deconv3_bn = tf.layers.batch_normalization(deconv3)

            deconv2 = tf.layers.conv2d_transpose(inputs=deconv3_bn, filters=128, kernel_size=(3, 3), padding='same',
                                                 activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            deconv2_bn = tf.layers.batch_normalization(deconv2)

            deconv1 = tf.layers.conv2d_transpose(inputs=deconv2_bn, filters=64, kernel_size=(3, 3), padding='same',
                                                 activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            deconv1_bn = tf.layers.batch_normalization(deconv1)

            deconv0 = tf.layers.conv2d_transpose(inputs=deconv1_bn, filters=3, kernel_size=(3, 3), padding='same',
                                                 activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        return deconv0




class test_Generator2():
    """
    MNIST_Discriminator2
    """
    def __init__(self):
        pass

    def make_generator_network(self, mask, reuse=False, isTrain=True):
        with tf.variable_scope("generator", reuse=reuse):
            input = mask
            # inputs_ = tf.placeholder(tf.float32, (None, 16, 16, 3), name="input")
            # targets_ = tf.placeholder(tf.float32, (None, 16, 16, 1), name="target")
            ### Encoder

            conv1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=(5, 5), padding='same', strides=2,
                                     activation=tf.nn.leaky_relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv1_bn = tf.layers.batch_normalization(conv1)

            conv2 = tf.layers.conv2d(inputs=conv1_bn, filters=128, kernel_size=(5, 5), padding='same', strides=2,
                                     activation=tf.nn.leaky_relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv2_bn = tf.layers.batch_normalization(conv2)

            conv3 = tf.layers.conv2d(inputs=conv2_bn, filters=256, kernel_size=(5, 5), padding='same', strides=2,
                                     activation=tf.nn.leaky_relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv3_bn = tf.layers.batch_normalization(conv3)

            conv4 = tf.layers.conv2d(inputs=conv3_bn, filters=512, kernel_size=(5, 5), padding='same', strides=2,
                                     activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv4_bn = tf.layers.batch_normalization(conv4)

            #fc4_reshape = tf.reshape(conv4_bn, shape=[-1,2048])

            deconv4 = tf.layers.conv2d_transpose(inputs=conv4_bn, filters=256, kernel_size=(5, 5), padding='same', strides=2,
                                     activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            deconv4_bn = tf.layers.batch_normalization(deconv4)

            ### Bottleneck

            # fc4_reshape = tf.reshape(conv3_bn, shape = [ -1,32*32*64])
            # fc4 = tf.layers.dense(fc4_reshape, units=1024)
            #
            # fc5 = tf.layers.dense(fc4, units=32*32*64)
            #
            # fc5_reshape = tf.reshape(fc5, shape=[-1, 32 , 32 , 64])


            ### Decoder

            deconv3 = tf.layers.conv2d_transpose(inputs=deconv4_bn, filters=128, kernel_size=(5, 5), padding='same', strides=2,
                                     activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            deconv3_bn = tf.layers.batch_normalization(deconv3)

            deconv2 = tf.layers.conv2d_transpose(inputs=deconv3_bn, filters=64, kernel_size=(5, 5), padding='same', strides=2,
                                     activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            deconv2_bn = tf.layers.batch_normalization(deconv2)


            deconv1 = tf.layers.conv2d_transpose(inputs=deconv2_bn, filters=3, kernel_size=(5, 5), padding='same', strides=2,
                                     activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        return deconv1
