#!/usr/bin/python3

from tensorflow.keras.layers import MaxPooling2D, Conv2D, Input, \
    BatchNormalization, UpSampling2D, Concatenate
from tensorflow.keras.models import Model


# TODO: Someone calls it small U-Net - check variations
def get_unet(nr_classes, nr_bands=12, nr_filters=16, bn=True, dilation_rate=1,
             tensor_shape=(256, 256)):
    """Create the U-Net architecture.

    :param nr_classes: number of classes to be predicted
    :param nr_bands: number of bands of intended input images
    :param nr_filters: base number of convolution filters (multiplied deeper
        in the model)
    :param bn: boolean saying whether to use batch normalization or not
    :param dilation_rate: convolution dilation rate
    :param tensor_shape: shape of the first two dimensions of input tensors
    :return: U-Net model
    """
    concat_layers = []

    inputs = Input((tensor_shape[0], tensor_shape[1], nr_bands))
    x = inputs

    for i in range(4):
        for j in range(2):
            x = Conv2D(nr_filters * (2 ** i),
                       (3, 3),
                       activation='relu',
                       padding='same',
                       dilation_rate=dilation_rate)(x)
            if bn:
                x = BatchNormalization()(x)

        concat_layers.append(x)
        x = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(x)

    for i in range(4, 0, -1):
        for j in range(2):
            x = Conv2D(nr_filters * (2 ** i),
                       (3, 3),
                       activation='relu',
                       padding='same',
                       dilation_rate=dilation_rate)(x)
            if bn:
                x = BatchNormalization()(x)

        x = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(x),
                                 concat_layers[i - 1]])

    conv9 = x

    for i in range(2):
        conv9 = Conv2D(nr_filters * 1,
                       (3, 3),
                       activation='relu',
                       padding='same',
                       dilation_rate=dilation_rate)(conv9)
        if bn:
            conv9 = BatchNormalization()(conv9)

    # number of classes
    conv10 = Conv2D(nr_classes,
                    (1, 1),
                    activation='softmax',
                    padding='same',
                    dilation_rate=dilation_rate)(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model
