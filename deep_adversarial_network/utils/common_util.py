"""
    Utility class for layer initialization
"""
import tensorflow as tf
import numpy as np


def noise(n_rows, n_cols):
    return np.random.normal(size=(n_rows, n_cols))

def xavier_init(size):
    in_dim = size[0] if len(size) == 1 else size[1]
    stddev = 1. / np.sqrt(float(in_dim))
    return tf.random_uniform(shape=size, minval=-stddev, maxval=stddev)

def images_to_vectors(images):
    return images.reshape(images.shape[0], 784)

def vectors_to_images(vectors):
    return vectors.reshape(vectors.shape[0], 28, 28, 1)


def fixed_padding(inputs, kernel_size):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer())
