"""
discriminator for mnist dataset
"""
import tensorflow as tf
from deep_adversarial_network.generator.base_generator import BaseGenerator
from deep_adversarial_network.utils.common_util import *

class MNIST_Generator1(BaseGenerator):
    """
    MNIST_Discriminator1
    """
    def __init__(self, *args, **kwargs):
        super(MNIST_Generator1, self).__init__(*args, **kwargs)

    def make_generator_variables(self):
        ## Generator

        # Layer 1 Variables
        self.G_W1 = tf.Variable(xavier_init([100, 256]))
        self.G_B1 = tf.Variable(xavier_init([256]))

        # Layer 2 Variables
        self.G_W2 = tf.Variable(xavier_init([256, 512]))
        self.G_B2 = tf.Variable(xavier_init([512]))

        # Layer 3 Variables
        self.G_W3 = tf.Variable(xavier_init([512, 1024]))
        self.G_B3 = tf.Variable(xavier_init([1024]))

        # Out Layer Variables
        self.G_W4 = tf.Variable(xavier_init([1024, 784]))
        self.G_B4 = tf.Variable(xavier_init([784]))

        # Store Variables in list
        G_var_list = [self.G_W1, self.G_B1, self.G_W2, self.G_B2, self.G_W3, self.G_B3, self.G_W4, self.G_B4]
        return  G_var_list


    def make_generator_network(self, z):

        l1 = tf.nn.leaky_relu(tf.matmul(z, self.G_W1) + self.G_B1, .2)
        l2 = tf.nn.leaky_relu(tf.matmul(l1, self.G_W2) + self.G_B2, .2)
        l3 = tf.nn.leaky_relu(tf.matmul(l2, self.G_W3) + self.G_B3, .2)
        out = tf.nn.tanh(tf.matmul(l3, self.G_W4) + self.G_B4)
        return out
