"""
 init class for all the generators for all the datasets
"""

from deep_adversarial_network.generator.test_generator import *

GENERATORS = {"test_generator1", "skip1", "skip2", "multi1", "multi2"}


def get_available_generators():
    """
    lists all the available discriminators
    :return: None
    """
    return sorted(GENERATORS)


def make_generator(name, *args, **kwargs):
    """
    creates the autoencoder based on the name
    :param name: string name of the autoencoder
    :param args: params for the autoenocoder object
    :param kwargs: params for the autoenocoder object
    :return: the autoencoder object
    """
    name = name.strip().lower()
    if not name in GENERATORS:
        raise ValueError("invalid autoencoder architecture: '{0}'".format(name))

    elif name == "test_generator1":
        return test_Generator1()
    elif name == "skip1":
        return Skip_Generator1()
    elif name == "skip2":
        return Skip_Generator2()
    elif name == "multi1":
        return  Multi_Generator1()
    elif name == "multi2":
        return  Multi_Generator2()
