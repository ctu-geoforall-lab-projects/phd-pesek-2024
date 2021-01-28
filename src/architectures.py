#!/usr/bin/python3

from tensorflow.keras.layers import MaxPooling2D, Conv2D, Input, UpSampling2D, \
    Concatenate, Dropout
from tensorflow.keras.models import Model

from cnn_lib import ConvBlock


# TODO: Someone calls it small U-Net - check variations
def get_unet(nr_classes, nr_bands=12, nr_filters=64, batch_norm=True,
             dilation_rate=1, tensor_shape=(256, 256), activation='relu',
             padding='same', dropout_rate_input=None, dropout_rate_hidden=None):
    """Create the U-Net architecture.

    For the original paper, see <https://arxiv.org/pdf/1505.04597.pdf>.
    The original architecture was enhanced by the option to perform dropout
    and batch normalization.

    :param nr_classes: number of classes to be predicted
    :param nr_bands: number of bands of intended input images
    :param nr_filters: base number of convolution filters (multiplied deeper
        in the model)
    :param batch_norm: boolean saying whether to use batch normalization or not
    :param dilation_rate: convolution dilation rate
    :param tensor_shape: shape of the first two dimensions of input tensors
    :param activation: activation function, such as tf.nn.relu, or string
        name of built-in activation function, such as 'relu'
    :param padding: 'valid' means no padding. 'same' results in padding
        evenly to the left/right or up/down of the input such that output
        has the same height/width dimension as the input
    :param dropout_rate_input: float between 0 and 1. Fraction of the input
        units of the input layer to drop
    :param dropout_rate_hidden: float between 0 and 1. Fraction of the input
        units of the hidden layers to drop
    :return: U-Net model
    """
    concat_layers = []

    # create input layer from the input tensor
    inputs = Input((tensor_shape[0], tensor_shape[1], nr_bands))

    if dropout_rate_input is not None:
        x = Dropout(rate=dropout_rate_input)(inputs)
    else:
        x = inputs

    # downsampling
    for i in range(4):
        block = ConvBlock(nr_filters * (2 ** i), (3, 3), activation, padding,
                          dilation_rate, dropout_rate=dropout_rate_hidden)
        x = block(x)
        concat_layers.append(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                         data_format='channels_last')(x)

    # upsampling
    for i in range(4, 0, -1):
        block = ConvBlock(nr_filters * (2 ** i), (3, 3), activation, padding,
                          dilation_rate, dropout_rate=dropout_rate_hidden)
        x = block(x)
        x = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(x),
                                 concat_layers[i - 1]])

    # the last upsampling without concatenation
    block = ConvBlock(nr_filters * 1, (3, 3), activation, padding,
                      dilation_rate, dropout_rate=dropout_rate_hidden)
    x = block(x)

    # softmax classifier head layer
    classes = Conv2D(nr_classes,
                     (1, 1),
                     activation='softmax',
                     padding=padding,
                     dilation_rate=dilation_rate)(x)

    model = Model(inputs=inputs, outputs=classes)

    return model
