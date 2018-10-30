"""
base class for discriminator creation
"""
import tensorflow as tf


class BaseDisciminator(object):

    def __init__(self, x):
        """
        initialize the parameters
        """
        super(BaseDisciminator, self).__init__()
        self.discriminator_variables = self.make_discriminator_variables()
        self.discriminator_network = self.make_discriminator_network(x)

    def make_discriminator_variables(self):
        """
        create the discriminator network
        :return: discriminator network variables
        """
        raise NotImplementedError('`make_discriminator_variables` is not implemented')

    def make_discriminator_network(self, x):
        """
        create the discriminator network
        :return: discriminator network output
        """
        raise NotImplementedError('`make_discriminator_network` is not implemented')