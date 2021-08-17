#!/usr/bin/python3

from tensorflow.keras.layers import MaxPooling2D, Conv2D, Input, UpSampling2D, \
    Concatenate, Dropout
from tensorflow.keras.models import Model

from cnn_lib import ConvBlock, MyMaxPooling, MyMaxUnpooling
from cnn_exceptions import ModelConfigError


def get_unet(nr_classes, nr_bands=12, nr_filters=64, batch_norm=True,
             dilation_rate=1, tensor_shape=(256, 256), activation='relu',
             padding='same', dropout_rate_input=None, dropout_rate_hidden=None):
    """Create the U-Net architecture.

    For the original paper, see <https://arxiv.org/pdf/1505.04597.pdf>.
    The original architecture was enhanced by the option to perform dropout
    and batch normalization and to specify padding (no padding in the
    original - cropping would be needed in such case).

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
    # check the reasonability of the architecture parameters
    if tensor_shape[0] % (2 ** 4) != 0 or tensor_shape[1] % (2 ** 4) != 0:
        raise ModelConfigError(
            'The tensor height and tensor width must be devidable by 32 for '
            'the U-Net architecture, but they are {} and {} respectively '
            'instead'.format(tensor_shape[0], tensor_shape[1]))

    # choose the activation function for the last layer
    if nr_classes == 2:
        activation_last = 'sigmoid'
    else:
        activation_last = 'softmax'

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
                          dilation_rate, dropout_rate=dropout_rate_hidden,
                          depth=2)
        x = block(x)
        concat_layers.append(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                         data_format='channels_last')(x)

    x = ConvBlock(nr_filters * (2 ** 4), (3, 3), activation, padding,
                  dilation_rate, dropout_rate=dropout_rate_hidden, depth=2)(x)

    # upsampling
    for i in range(3, -1, -1):
        x = UpSampling2D(size=(2, 2))(x)
        conv2 = Conv2D(nr_filters * (2 ** i), (2, 2), padding=padding,
                       dilation_rate=dilation_rate)
        # concatenate the upsampled weights with the corresponding ones from
        # the contracting path
        x = Concatenate(axis=3)([conv2(x), concat_layers[i]])
        block = ConvBlock(nr_filters * (2 ** i), (3, 3), activation, padding,
                          dilation_rate, dropout_rate=dropout_rate_hidden,
                          depth=2)
        x = block(x)

    # softmax classifier head layer
    classes = Conv2D(nr_classes,
                     (1, 1),
                     activation=activation_last,
                     padding=padding,
                     dilation_rate=dilation_rate)(x)

    model = Model(inputs=inputs, outputs=classes)

    return model

def get_segnet(nr_classes, nr_bands=12, nr_filters=64, batch_norm=True,
               dilation_rate=1, tensor_shape=(256, 256), activation='relu',
               padding='same', dropout_rate_input=None,
               dropout_rate_hidden=None):
    """Create the SegNet architecture.

    For the original paper, see <https://arxiv.org/pdf/1511.00561.pdf>.
    The original architecture was enhanced by the option to perform dropout
    and batch normalization and to specify padding.

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
    :return: SegNet model
    """
    # check the reasonability of the architecture parameters
    if tensor_shape[0] % (2 ** 4) != 0 or tensor_shape[1] % (2 ** 4) != 0:
        raise ModelConfigError(
            'The tensor height and tensor width must be devidable by 32 for '
            'the SegNet architecture, but they are {} and {} respectively '
            'instead'.format(tensor_shape[0], tensor_shape[1]))

    # choose the activation function for the last layer
    if nr_classes == 2:
        activation_last = 'sigmoid'
    else:
        activation_last = 'softmax'

    pooling_indices = []

    # create input layer from the input tensor
    inputs = Input((tensor_shape[0], tensor_shape[1], nr_bands))

    if dropout_rate_input is not None:
        x = Dropout(rate=dropout_rate_input)(inputs)
    else:
        x = inputs

    # downsampling
    for i in range(2):
        # blocks of the depth 2
        block = ConvBlock(nr_filters * (2 ** i), (3, 3), activation, padding,
                          dilation_rate, dropout_rate=dropout_rate_hidden,
                          depth=2)
        x = block(x)
        x, ind = MyMaxPooling(pool_size=(2, 2), strides=(2, 2),
                              data_format='channels_last')(x)
        pooling_indices.append(ind)

    for i in range(2, 5):
        # blocks of the depth 3
        block = ConvBlock(nr_filters * (2 ** i), (3, 3), activation, padding,
                          dilation_rate, dropout_rate=dropout_rate_hidden,
                          depth=3)
        x = block(x)
        x, ind = MyMaxPooling(pool_size=(2, 2), strides=(2, 2),
                              data_format='channels_last')(x)
        pooling_indices.append(ind)

    # upsampling
    for i in range(4, 1, -1):
        # blocks of the depth 3
        # upsampling with shared indices
        x = MyMaxUnpooling(pool_size=(2, 2))(x, pooling_indices[i])
        block = ConvBlock(nr_filters * (2 ** i), (3, 3), activation,
                          padding, dilation_rate,
                          dropout_rate=dropout_rate_hidden, depth=2)
        x = block(x)
        block = ConvBlock(nr_filters * (2 ** (i - 1)), (3, 3), activation,
                          padding, dilation_rate,
                          dropout_rate=dropout_rate_hidden, depth=1)
        x = block(x)

    # a block of the depth 2
    x = MyMaxUnpooling(pool_size=(2, 2))(x, pooling_indices[1])
    block = ConvBlock(nr_filters * (2 ** 1), (3, 3), activation,
                      padding, dilation_rate,
                      dropout_rate=dropout_rate_hidden, depth=1)
    x = block(x)
    block = ConvBlock(nr_filters * (2 ** 0), (3, 3), activation,
                      padding, dilation_rate,
                      dropout_rate=dropout_rate_hidden, depth=1)
    x = block(x)

    # a block of the depth 1
    # the paper states depth two and then softmax, but I believe that this
    # should do the same trick
    x = MyMaxUnpooling(pool_size=(2, 2))(x, pooling_indices[0])
    block = ConvBlock(nr_filters * (2 ** 0), (3, 3), activation,
                      padding, dilation_rate,
                      dropout_rate=dropout_rate_hidden, depth=1)
    x = block(x)

    # softmax classifier head layer
    classes = Conv2D(nr_classes,
                     (1, 1),
                     activation=activation_last,
                     padding=padding,
                     dilation_rate=dilation_rate)(x)

    model = Model(inputs=inputs, outputs=classes)

    return model
