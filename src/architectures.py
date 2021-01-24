#!/usr/bin/python3

from tensorflow.keras.layers import MaxPooling2D, Conv2D, Input, UpSampling2D, \
    Concatenate
from tensorflow.keras.models import Model

from cnn_lib import ConvBlock


# TODO: Someone calls it small U-Net - check variations
def get_unet(nr_classes, nr_bands=12, nr_filters=16, batch_norm=True,
             dilation_rate=1, tensor_shape=(256, 256)):
    """Create the U-Net architecture.

    :param nr_classes: number of classes to be predicted
    :param nr_bands: number of bands of intended input images
    :param nr_filters: base number of convolution filters (multiplied deeper
        in the model)
    :param batch_norm: boolean saying whether to use batch normalization or not
    :param dilation_rate: convolution dilation rate
    :param tensor_shape: shape of the first two dimensions of input tensors
    :return: U-Net model
    """
    concat_layers = []

    # create input layer from the input tensor
    inputs = Input((tensor_shape[0], tensor_shape[1], nr_bands))
    x = inputs

    # downsampling
    for i in range(4):
        block = ConvBlock(nr_filters * (2 ** i), (3, 3), 'relu', 'same',
                          dilation_rate)
        x = block(x)
        concat_layers.append(x)
        x = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(x)

    # upsampling
    for i in range(4, 0, -1):
        block = ConvBlock(nr_filters * (2 ** i), (3, 3), 'relu', 'same',
                          dilation_rate)
        x = block(x)
        x = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(x),
                                 concat_layers[i - 1]])

    # the last upsampling without concatenation
    block = ConvBlock(nr_filters * 1, (3, 3), 'relu', 'same',
                      dilation_rate)
    x = block(x)

    # softmax classifier head layer
    classes = Conv2D(nr_classes,
                     (1, 1),
                     activation='softmax',
                     padding='same',
                     dilation_rate=dilation_rate)(x)

    model = Model(inputs=inputs, outputs=classes)

    return model
