#!/usr/bin/python3

import os

import numpy as np
import tensorflow as tf

from osgeo import gdal
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, \
    Activation, Dropout

from data_preparation import generate_dataset_structure


# TODO: check tf.keras.preprocessing.image.ImageDataGenerator
# TODO: check keras.utils.Sequence
# TODO: Does not really augment, does it?
# TODO: support onehot_encode boolean parameter
class AugmentGenerator:
    """Data generator."""

    def __init__(self, data_dir, batch_size=5, operation='train',
                 nr_bands=12, tensor_shape=(256, 256),
                 force_dataset_generation=False, fit_memory=False):
        """Initialize the generator.

        :param data_dir: path to the directory containing images
        :param batch_size: the number of samples that will be propagated
            through the network at once
        :param operation: either 'train' or 'val'
        :param nr_bands: number of bands of intended input images
        :param tensor_shape: shape of the first two dimensions of input tensors
        :param force_dataset_generation: boolean to force the dataset
            structure generation
        :param fit_memory: boolean to load the entire dataset into memory
            instead of opening new files with each request
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
        self.image_generator = self.numpy_generator(
            images_dir, False, batch_size, fit_memory)
        self.mask_generator = self.numpy_generator(
            masks_dir, False, batch_size, fit_memory)

        # create variables holding number of samples
        self.nr_samples = len(os.listdir(images_dir))

    def __call__(self, id2code, seed=1):
        """Generate batches of data.

        Note: tf.data.Dataset.from_generator() seemed to be useful and maybe
        could speed up the process little bit , but it seemed not to work
        properly when __call__ takes arguments.

        :param id2code: dictionary mapping label ids to their codes
        :param seed: the generator seed
        :return: yielded tuple of batch-sized np stacks of validation images and
            masks
        """
        while True:
            x1i = next(self.image_generator)
            x2i = next(self.mask_generator)

            # TODO: have seen the following somewhere - check
            # mask_encoded = [onehot_encode(x2i[0][x, :, :, :], id2code) for x in
            #                 range(x2i[0].shape[0])]

            # one hot encode masks
            mask_encoded = [
                self.onehot_encode(x2i[x, :, :, :], id2code) for x in
                range(x2i.shape[0])]

            yield x1i, np.asarray(mask_encoded)

    def numpy_generator(self, data_dir, rescale=False, batch_size=5,
                        fit_memory=False):
        """Generate batches of images.

        :param data_dir: path to the directory containing images
        :param rescale: boolean saying whether to rescale images or not
            (rescaling is a division by 255)
        :param batch_size: the number of samples that will be propagated through
            the network at once
        :param fit_memory: boolean to load the entire dataset into memory
            instead of opening new files with each request
        :return: yielded batch-sized np stack of images
        """
        index = 1
        batch = []

        # list of files from which the batches will be created
        files_list = sorted(os.listdir(data_dir))

        if fit_memory is True:
            # fit the dataset into memory
            source_list = []
            for file in files_list:
                image = self.transpose_image(data_dir, file, rescale)

                # add the image to the source list
                source_list.append(image)
        else:
            source_list = files_list

        while True:
            for source in source_list:
                if fit_memory is True:
                    image = source
                else:
                    image = self.transpose_image(data_dir, source, rescale)

                # add the image to the batch
                batch.append(image)

                if index % batch_size == 0:
                    # batch created, return it
                    yield np.stack(batch)
                    batch = []

                index += 1

    @staticmethod
    def transpose_image(data_dir, image_name, rescale):
        """Open an image and transpose it to (1, 2, 0).

        :param data_dir: path to the directory containing images
        :param image_name: name of the image file in the data dir
        :param rescale: boolean saying whether to rescale images or not
            (rescaling is a division by 255)
        :return: the transposed image as a numpy array
        """
        image = gdal.Open(os.path.join(data_dir, image_name))
        image_array = image.ReadAsArray()

        # GDAL reads masks as having no third dimension
        # (we want it to be equal to one)
        if image_array.ndim == 2:
            transposed = np.expand_dims(image_array, -1)
        else:
            # move the batch to be the last dimension
            transposed = np.moveaxis(image.ReadAsArray(), 0, -1)

        if rescale:
            transposed = 1. / 255 * transposed

        image = None

        return transposed

    @staticmethod
    def onehot_encode(orig_image, colormap):
        """Encode input images into one hot ones.

        Unfortunately, keras.utils.to_categorical cannot be used because our
        classes are not consecutive.

        :param orig_image: original image
        :param colormap: dictionary mapping label ids to their codes
        :return: One hot encoded image of dimensions
            (height x width x num_classes)
        """
        num_classes = len(colormap)
        shape = orig_image.shape[:2] + (num_classes,)
        encoded_image = np.empty(shape, dtype=np.uint8)

        # reshape to the shape used inside the onehot matrix
        reshaped = orig_image.reshape((-1, 1))

        for i, cls in enumerate(colormap):
            all_ax = np.all(reshaped == colormap[i], axis=1)
            encoded_image[:, :, i] = all_ax.reshape(shape[:2])

        return encoded_image



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

    def __init__(self, nr_filters=64, kernel_size=(3, 3), activation='relu',
                 padding='same', dilation_rate=1, batch_norm=True,
                 dropout_rate=None, **kwargs):
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
        :param dropout_rate: float between 0 and 1. Fraction of the input
            units of convolutional layers to drop
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
        self.dropout_rate = dropout_rate
        self.kwargs = kwargs

        # instantiate layers of the conv block
        self.conv_layer1 = Conv2D(nr_filters, kernel_size, padding=padding,
                                  dilation_rate=dilation_rate)
        self.activation1 = Activation(activation)
        self.batch_norm1 = BatchNormalization()
        self.dropout1 = Dropout(rate=dropout_rate)
        self.conv_layer2 = Conv2D(nr_filters, kernel_size, padding=padding,
                                  dilation_rate=dilation_rate)
        self.dropout2 = Dropout(rate=dropout_rate)
        self.activation2 = Activation(activation)
        self.batch_norm2 = BatchNormalization()

    def call(self, x, mask=None):
        """Perform the logic of applying the layer to the input tensors.

        :param x: input tensor
        :param mask: boolean tensor encoding masked timesteps in the input,
            used in RNN layers (currently not used)
        :return: output layer of the convolutional block
        """
        # the first layer of the block
        x = self.conv_layer1(x)
        if self.dropout_rate is not None:
            x = self.dropout1(x)
        x = self.activation1(x)
        if self.batch_norm is True:
            x = self.batch_norm1(x)

        # the second layer of the block
        x = self.conv_layer2(x)
        if self.dropout_rate is not None:
            x = self.dropout2(x)
        x = self.activation2(x)
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
                      dropout_rate=self.dropout_rate,
                      **self.kwargs)

        return config
