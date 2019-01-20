from deep_adversarial_network.losses.custom_vgg16 import *


def perceptual_loss(batch_size, inputs, outputs):
    """
    Perceptual Loss
    :param batch_size:
    :param inputs:
    :param outputs:
    :return:
    """
    direc = os.getcwd()
    data_dict = loadWeightsData(direc + '/deep_adversarial_network/weights/vgg16.npy')
    #tf.flags.DEFINE_integer("batch_size", 5, "Batch size during training")
    vgg_c = custom_Vgg16(inputs, data_dict=data_dict)
    feature_ = [vgg_c.conv1_2, vgg_c.conv2_2, vgg_c.conv3_3, vgg_c.conv4_3, vgg_c.conv5_3]

    vgg = custom_Vgg16(outputs, data_dict=data_dict)
    feature = [vgg.conv1_2, vgg.conv2_2, vgg.conv3_3, vgg.conv4_3, vgg.conv5_3]

    loss_f = tf.zeros(batch_size, tf.float32)
    for f, f_ in zip(feature, feature_):
        loss_f += tf.reduce_mean(tf.subtract(f, f_) ** 2)

    return loss_f
