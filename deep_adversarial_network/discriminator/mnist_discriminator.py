"""
discriminator for mnist dataset
"""
import tensorflow as tf
from deep_adversarial_network.discriminator.base_discriminator import BaseDisciminator
from deep_adversarial_network.utils.common_util import *


class MNIST_Discriminator1(BaseDisciminator):
    """
    MNIST_Discriminator1
    """
    def __init__(self, *args, **kwargs):
        super(MNIST_Discriminator1, self).__init__(*args, **kwargs)

    def make_discriminator_variables(self):
        ## Discriminator

        # Layer 1 Variables
        self.D_W1 = tf.Variable(xavier_init([784, 1024]))
        self.D_B1 = tf.Variable(xavier_init([1024]))

        # Layer 2 Variables
        self.D_W2 = tf.Variable(xavier_init([1024, 512]))
        self.D_B2 = tf.Variable(xavier_init([512]))

        # Layer 3 Variables
        self.D_W3 = tf.Variable(xavier_init([512, 256]))
        self.D_B3 = tf.Variable(xavier_init([256]))

        # Out Layer Variables
        self.D_W4 = tf.Variable(xavier_init([256, 1]))
        self.D_B4 = tf.Variable(xavier_init([1]))

        # Store Variables in list
        D_var_list = [self.D_W1, self.D_B1, self.D_W2, self.D_B2, self.D_W3, self.D_B3, self.D_W4, self.D_B4]
        return  D_var_list

    def make_discriminator_network(self, x):

        l1 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(x, self.D_W1) + self.D_B1, .2), .3)
        l2 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(l1, self.D_W2) + self.D_B2, .2), .3)
        l3 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(l2, self.D_W3) + self.D_B3, .2), .3)
        out = tf.matmul(l3, self.D_W4) + self.D_B4
        return out
