"""
 init class for all the discrimina for all the datasets
"""

from deep_adversarial_network.discriminator.test_discriminator import *

DISCRIMINATORS = {"test_discriminator1", "resnet", "patch"}


def get_available_discriminators():
    """
    lists all the available discriminators
    :return: None
    """
    return sorted(DISCRIMINATORS)


def make_discriminator(name, *args, **kwargs):
    """
    creates the autoencoder based on the name
    :param name: string name of the autoencoder
    :param args: params for the autoenocoder object
    :param kwargs: params for the autoenocoder object
    :return: the autoencoder object
    """
    name = name.strip().lower()
    if not name in DISCRIMINATORS:
        raise ValueError("invalid discriminator architecture: '{0}'".format(name))

    elif name == "test_discriminator1":
        return test_Discriminator1()

    elif name == "resnet":
        return Resnet_Discriminator()

    elif name == "patch":
        return Patch_Discriminator()
