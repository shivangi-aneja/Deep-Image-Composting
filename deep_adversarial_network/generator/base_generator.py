"""
base class for generator creation
"""
import tensorflow as tf


class BaseGenerator(object):
    def __init__(self, z):
        """
        initialize the parameters
        """
        super(BaseGenerator, self).__init__()
        self.generator_variables = self.make_generator_variables()
        self.generator_network = self.make_generator_network(z)


    def make_generator_variables(self):
        """
        create the discriminator network
        :return: discriminator network variables
        """
        raise NotImplementedError('`make_generator_variables` is not implemented')

    def make_generator_network(self, z):
        """
        create the generator network
        :return: generator network output and variables
        """
        raise NotImplementedError('`make_generator_network` is not implemented')