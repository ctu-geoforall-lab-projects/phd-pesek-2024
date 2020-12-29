#!/usr/bin/python3

# TODO: change concatenate to Concatenate
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Input, \
    BatchNormalization, UpSampling2D, concatenate
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
    inputs = Input((tensor_shape[0], tensor_shape[1], nr_bands))

    # TODO: avoid duplicity - make more systematic
    conv1 = Conv2D(nr_filters * 1, (3, 3), activation='relu', padding='same',
                   dilation_rate=dilation_rate)(inputs)
    if bn:
        conv1 = BatchNormalization()(conv1)

    conv1 = Conv2D(nr_filters * 1, (3, 3), activation='relu', padding='same',
                   dilation_rate=dilation_rate)(conv1)
    if bn:
        conv1 = BatchNormalization()(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv1)

    conv2 = Conv2D(nr_filters * 2, (3, 3), activation='relu', padding='same',
                   dilation_rate=dilation_rate)(pool1)
    if bn:
        conv2 = BatchNormalization()(conv2)

    conv2 = Conv2D(nr_filters * 2, (3, 3), activation='relu', padding='same',
                   dilation_rate=dilation_rate)(conv2)
    if bn:
        conv2 = BatchNormalization()(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv2)

    conv3 = Conv2D(nr_filters * 4, (3, 3), activation='relu', padding='same',
                   dilation_rate=dilation_rate)(pool2)
    if bn:
        conv3 = BatchNormalization()(conv3)

    conv3 = Conv2D(nr_filters * 4, (3, 3), activation='relu', padding='same',
                   dilation_rate=dilation_rate)(conv3)
    if bn:
        conv3 = BatchNormalization()(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv3)

    conv4 = Conv2D(nr_filters * 8, (3, 3), activation='relu', padding='same',
                   dilation_rate=dilation_rate)(pool3)
    if bn:
        conv4 = BatchNormalization()(conv4)

    conv4 = Conv2D(nr_filters * 8, (3, 3), activation='relu', padding='same',
                   dilation_rate=dilation_rate)(conv4)
    if bn:
        conv4 = BatchNormalization()(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv4)

    conv5 = Conv2D(nr_filters * 16, (3, 3), activation='relu', padding='same',
                   dilation_rate=dilation_rate)(pool4)
    if bn:
        conv5 = BatchNormalization()(conv5)

    conv5 = Conv2D(nr_filters * 16, (3, 3), activation='relu', padding='same',
                   dilation_rate=dilation_rate)(conv5)
    if bn:
        conv5 = BatchNormalization()(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)

    conv6 = Conv2D(nr_filters * 8, (3, 3), activation='relu', padding='same',
                   dilation_rate=dilation_rate)(up6)  # (conv4)  # (up6)
    if bn:
        conv6 = BatchNormalization()(conv6)

    conv6 = Conv2D(nr_filters * 8, (3, 3), activation='relu', padding='same',
                   dilation_rate=dilation_rate)(conv6)
    if bn:
        conv6 = BatchNormalization()(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)

    conv7 = Conv2D(nr_filters * 4, (3, 3), activation='relu', padding='same',
                   dilation_rate=dilation_rate)(up7)  # (conv3)  # (up7)
    if bn:
        conv7 = BatchNormalization()(conv7)

    conv7 = Conv2D(nr_filters * 4, (3, 3), activation='relu', padding='same',
                   dilation_rate=dilation_rate)(conv7)
    if bn:
        conv7 = BatchNormalization()(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)

    conv8 = Conv2D(nr_filters * 2, (3, 3), activation='relu', padding='same',
                   dilation_rate=dilation_rate)(up8)
    if bn:
        conv8 = BatchNormalization()(conv8)

    conv8 = Conv2D(nr_filters * 2, (3, 3), activation='relu', padding='same',
                   dilation_rate=dilation_rate)(conv8)
    if bn:
        conv8 = BatchNormalization()(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)

    conv9 = Conv2D(nr_filters * 1, (3, 3), activation='relu', padding='same',
                   dilation_rate=dilation_rate)(up9)
    if bn:
        conv9 = BatchNormalization()(conv9)

    conv9 = Conv2D(nr_filters * 1, (3, 3), activation='relu', padding='same',
                   dilation_rate=dilation_rate)(conv9)
    if bn:
        conv9 = BatchNormalization()(conv9)

    # number of classes
    conv10 = Conv2D(nr_classes, (1, 1), activation='softmax',
                    padding='same', dilation_rate=dilation_rate)(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model
