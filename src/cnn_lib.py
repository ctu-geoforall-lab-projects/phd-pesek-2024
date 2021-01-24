#!/usr/bin/python3

import os
import rasterio

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization

from data_preparation import generate_dataset_structure


# TODO: check keras.utils.to_categorical
def onehot_encode(orig_image, colormap):
    """Encode input images into one hot ones.

    :param orig_image: original image
    :param colormap: dictionary mapping label ids to their codes
    :return: One hot encoded image of dimensions (height x width x num_classes)
    """
    num_classes = len(colormap)
    shape = orig_image.shape[:2] + (num_classes,)
    # TODO: Test with np.empty, np.uint8
    encoded_image = np.zeros(shape, dtype=np.int8)
    for i, cls in enumerate(colormap):
        tup = (-1, 1)
        resh = orig_image.reshape(tup)
        eq = resh == colormap[i]
        all_ax = np.all(eq, axis=1)
        encoded_image[:, :, i] = all_ax.reshape(shape[:2])

    return encoded_image


# TODO: get rid of the rasterio dependency
def rasterio_generator(data_dir, rescale=False, batch_size=5):
    """Generate batches of images.

    :param data_dir: path to the directory containing images
    :param rescale: boolean saying whether to rescale images or not
        (rescaling is a division by 255)
    :param batch_size: the number of samples that will be propagated through
        the network at once
    :return: yielded batch-sized np stack of images
    """
    index = 1
    batch = []
    while True:
        for file in sorted(os.listdir(data_dir)):
            a = rasterio.open(os.path.join(data_dir, file))
            q = a.read()
            q = np.transpose(q, (1, 2, 0))
            if rescale:
                q = 1. / 255 * q

            batch.append(q)

            if index % batch_size == 0:
                yield np.stack(batch)
                batch = []

            index += 1


# TODO: check tf.data.Dataset.from_generator()
# TODO: check tf.keras.preprocessing.image.ImageDataGenerator
# TODO: check keras.utils.Sequence
# TODO: Does not really augment, does it?
# TODO: support onehot_encode boolean parameter
class AugmentGenerator:
    """Data generator."""

    def __init__(self, data_dir, batch_size=5, operation='train',
                 nr_bands=12, tensor_shape=(256, 256),
                 force_dataset_generation=False):
        """

        :param data_dir: path to the directory containing images
        :param batch_size: the number of samples that will be propagated
            through the network at once
        :param operation: either 'train' or 'val'
        :param nr_bands: number of bands of intended input images
        :param tensor_shape: shape of the first two dimensions of input tensors
        :param force_dataset_generation: boolean to force the dataset
            structure generation
        """
        if operation not in ('train', 'val'):
            raise AttributeError('Only values "train" and "val" supported as '
                                 'operation. "{}" was given'.format(operation))

        # TODO: the operation directory seems redundant
        images_dir = os.path.join(
            data_dir, '{}_images'.format(operation), operation)
        masks_dir = os.path.join(
            data_dir, '{}_masks'.format(operation), operation)
        # generate the dataset structure if not generated
        do_exist = [os.path.isdir(i) is True for i in (images_dir, masks_dir)]
        if force_dataset_generation or not all(do_exist):
            generate_dataset_structure(data_dir, nr_bands, tensor_shape)

        # create generators
        self.image_generator = rasterio_generator(
            images_dir, False, batch_size)
        self.mask_generator = rasterio_generator(
            masks_dir, False, batch_size)

        # create variables holding number of samples
        self.nr_samples = len(os.listdir(images_dir))

    def __call__(self, id2code, seed=1):
        """Generate batches of data.

        :param id2code: dictionary mapping label ids to their codes
        :param seed: the generator seed
        :return: yielded tuple of batch-sized np stacks of validation images and
            masks
        """
        while True:
            x1i = next(self.image_generator)
            x2i = next(self.mask_generator)

            # TODO: has seen the following somewhere - check
            # mask_encoded = [onehot_encode(x2i[0][x, :, :, :], id2code) for x in
            #                 range(x2i[0].shape[0])]

            # one hot encode masks
            mask_encoded = [onehot_encode(x2i[x, :, :, :], id2code) for x in
                            range(x2i.shape[0])]

            yield x1i, np.asarray(mask_encoded)


def categorical_dice(ground_truth_onehot, predictions, weights=1):
    """Compute the Sorensen-Dice loss.

    :param ground_truth_onehot: onehot ground truth labels
        (batch_size, img_height, img_width, nr_classes)
    :param predictions: predictions from the last layer of the CNN
        (batch_size, img_height, img_width, nr_classes)
    :param weights: weights for individual classes
        (number-of-classes-long vector)
    :return: dice loss value averaged for all classes
    """
    loss = categorical_tversky(ground_truth_onehot, predictions, 0.5, 0.5,
                               weights)

    return loss


def categorical_tversky(ground_truth_onehot, predictions, alpha=0.5,
                        beta=0.5, weights=1):
    """Compute the Tversky loss.

    alpha == beta == 0.5 -> Dice loss
    alpha == beta == 1 -> Tanimoto coefficient/loss

    :param ground_truth_onehot: onehot ground truth labels
        (batch_size, img_height, img_width, nr_classes)
    :param predictions: predictions from the last layer of the CNN
        (batch_size, img_height, img_width, nr_classes)
    :param alpha: magnitude of penalties for false positives
    :param beta: magnitude of penalties for false negatives
    :param weights: weights for individual classes
        (number-of-classes-long vector)
    :return: dice loss value averaged for all classes
    """
    weight_tensor = tf.constant(weights, dtype=tf.float32)
    predictions = tf.cast(predictions, tf.float32)
    ground_truth_onehot = tf.cast(ground_truth_onehot, tf.float32)

    # compute true positives, false negatives and false positives
    true_pos = ground_truth_onehot * predictions
    false_neg = ground_truth_onehot * (1. - predictions)
    false_pos = (1. - ground_truth_onehot) * predictions

    # compute Tversky coefficient
    numerator = true_pos
    numerator = tf.reduce_sum(numerator, axis=(1, 2))
    denominator = true_pos + alpha * false_neg + beta * false_pos
    denominator = tf.reduce_sum(denominator, axis=(1, 2))
    tversky = numerator / denominator

    # reduce mean for batches
    tversky = tf.reduce_mean(tversky, axis=0)

    # reduce mean for classes and multiply them by weights
    loss = 1 - tf.reduce_mean(weight_tensor * tversky)

    return loss


class ConvBlock(Layer):
    """TF Keras layer overriden to represent a convolutional block in U-Net."""

    def __init__(self, nr_filters=16, kernel_size=(3, 3), activation='relu',
                 padding='same', dilation_rate=1, batch_norm=True, **kwargs):
        """Create a block of two convolutional layers.

        Each of them could be followed by a batch normalization layer.

        :param nr_filters: number of convolution filters
        :param kernel_size: an integer or tuple/list of 2 integers, specifying
            the height and width of the 2D convolution window
        :param activation: activation function, such as tf.nn.relu, or string
            name of built-in activation function, such as 'relu'
        :param padding: 'valid' means no padding. 'same' results in padding
            evenly to the left/right or up/down of the input such that output
            has the same height/width dimension as the input
        :param dilation_rate: convolution dilation rate
        :param batch_norm: boolean saying whether to use batch normalization
            or not
        :param kwargs: supplementary kwargs for the parent __init__()
        """
        super(ConvBlock, self).__init__(**kwargs)

        # set init parameters to member variables
        self.nr_filters = nr_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.batch_norm = batch_norm
        self.kwargs = kwargs

        # instantiate layers of the conv block
        self.conv_layer1 = Conv2D(nr_filters,
                       kernel_size,
                       activation=activation,
                       padding=padding,
                       dilation_rate=dilation_rate)
        self.batch_norm1 = BatchNormalization()
        self.conv_layer2 = Conv2D(nr_filters,
                       kernel_size,
                       activation=activation,
                       padding=padding,
                       dilation_rate=dilation_rate)
        self.batch_norm2 = BatchNormalization()

    def call(self, x, training=True, mask=None):
        """Perform the logic of applying the layer to the input tensors.

        :param x: input tensor
        :param training: boolean saying whether the call is in inference mode
            or training mode (currenlty not used)
        :param mask: boolean tensor encoding masked timesteps in the input,
            used in RNN layers (currently not used)
        :return: output layer of the convolutional block
        """
        # the first layer of the block
        x = self.conv_layer1(x)
        if self.batch_norm is True:
            x = self.batch_norm1(x)

        # the second layer of the block
        x = self.conv_layer2(x)
        if self.batch_norm is True:
            x = self.batch_norm2(x)

        return x

    def get_config(self):
        """Return the configuration of the convolutional block.

        Allows later reinstantiation (without its trained weights) from this
        configuration. It does not include connectivity information, nor the
        layer class name.

        :return: the configuration dictionary of the convolutional block
        """
        config = super(ConvBlock, self).get_config()

        config.update(nr_filters=self.nr_filters,
                      kernel_size=self.kernel_size,
                      activation=self.activation,
                      padding=self.padding,
                      dilation_rate=self.dilation_rate,
                      batch_norm=self.batch_norm,
                      **self.kwargs)

        return config
