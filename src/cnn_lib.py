#!/usr/bin/python3

import os

import numpy as np
import tensorflow as tf

from osgeo import gdal
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, \
    Activation, Dropout
from tensorflow.keras import backend as keras

from data_preparation import generate_dataset_structure
from cnn_exceptions import LayerDefinitionError


class AugmentGenerator:
    """Data generator."""

    def __init__(self, data_dir, batch_size=5, operation='train',
                 nr_bands=12, tensor_shape=(256, 256),
                 force_dataset_generation=False, fit_memory=False,
                 augment=False, onehot_encode=True, val_set_pct=0.2,
                 filter_by_class=None):
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
        :param augment: boolean saying whether to augment the dataset or not
        :param onehot_encode: boolean to onehot-encode masks during training
        :param val_set_pct: percentage of the validation images in the dataset
        :param filter_by_class: classes of interest (for the case of dataset
            generation - if specified, only samples containing at least one of
            them will be created)
        """
        if operation not in ('train', 'val'):
            raise AttributeError('Only values "train" and "val" supported as '
                                 'operation. "{}" was given'.format(operation))

        images_dir = os.path.join(
            data_dir, '{}_images'.format(operation))
        masks_dir = os.path.join(
            data_dir, '{}_masks'.format(operation))
        # generate the dataset structure if not generated
        do_exist = [os.path.isdir(i) is True for i in (images_dir, masks_dir)]
        if force_dataset_generation is True or all(do_exist) is False:
            generate_dataset_structure(data_dir, nr_bands, tensor_shape,
                                       val_set_pct, filter_by_class)

        # create variables useful throughout the entire class
        self.nr_samples = len(os.listdir(images_dir))
        self.batch_size = batch_size
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.fit_memory = fit_memory
        self.augment = augment
        self.perform_onehot_encoding = onehot_encode

    def __call__(self, id2code, seed=1):
        """Generate batches of data.

        :param id2code: dictionary mapping label ids to their codes
        :param seed: the generator seed (unfortunately, the seed does not
            work properly in tensorflow, therefore it does not do what is
            expected when augment is set to True)
        :return: yielded tuple of batch-sized np stacks of validation images and
            masks
        """
        if self.augment is False:
            return self.generate_numpy(id2code)
        else:
            return self.generate_augmented(id2code, seed)

    def generate_numpy(self, id2code):
        """Generate batches of data using our own numpy generator.

        Note: tf.data.Dataset.from_generator() seemed to be useful and maybe
        could speed up the process little bit , but it seemed not to work
        properly when __call__ takes arguments.

        :param id2code: dictionary mapping label ids to their codes
        :return: yielded tuple of batch-sized np stacks of validation images and
            masks
        """
        # create generators
        image_generator = self.numpy_generator(
            self.images_dir, False, self.batch_size, self.fit_memory)
        mask_generator = self.numpy_generator(
            self.masks_dir, False, self.batch_size, self.fit_memory)

        while True:
            x1i = next(image_generator)
            x2i = next(mask_generator)

            if self.perform_onehot_encoding is True:
                # one hot encode masks
                x2i = [
                    self.onehot_encode(x2i[x, :, :, :], id2code) for x in
                    range(x2i.shape[0])]

            yield x1i, np.asarray(x2i)

    def generate_augmented(self, id2code, seed):
        """Generate batches of data using TF Keras augmenting class.

        :param id2code: dictionary mapping label ids to their codes
        :param seed: the generator seed
        :return: TF Keras ImageDataGenerator holding our data
        """
        images = np.stack(self.get_transposed_images(self.images_dir, False))
        masks = np.stack(self.get_transposed_images(self.masks_dir, False))

        if self.perform_onehot_encoding is True:
            # one hot encode masks
            masks = [
                self.onehot_encode(masks[x, :, :, :], id2code) for x in
                range(masks.shape[0])]

        datagen = ImageDataGenerator(rotation_range=180, shear_range=0.2,
                                     horizontal_flip=True, vertical_flip=True)

        datagen.fit(images, seed=seed, augment=True)

        return datagen.flow(images, np.asarray(masks), seed=seed,
                            batch_size=self.batch_size, shuffle=True)

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
        if fit_memory is True:
            # fit the dataset into memory
            source_list = self.get_transposed_images(data_dir, rescale)
        else:
            # list of files from which the batches will be created
            source_list = sorted(os.listdir(data_dir))

        index = 1
        batch = []

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

    def get_transposed_images(self, data_dir, rescale):
        """Get list of transposed images.

        :param data_dir: path to the directory containing images
        :param rescale: boolean saying whether to rescale images or not
            (rescaling is a division by 255)
        :return: list of transposed numpy matrices representing images in
            the dataset
        """
        # list of files from which the dataset will be created
        files_list = sorted(os.listdir(data_dir))

        images_list = [self.transpose_image(data_dir, file, rescale) for file in
                       files_list]

        return images_list

    @staticmethod
    def transpose_image(data_dir, image_name, rescale):
        """Open an image and transpose it to (1, 2, 0).

        :param data_dir: path to the directory containing images
        :param image_name: name of the image file in the data dir
        :param rescale: boolean saying whether to rescale images or not
            (rescaling is a division by 255)
        :return: the transposed image as a numpy array
        """
        image = gdal.Open(os.path.join(data_dir, image_name), gdal.GA_ReadOnly)
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


# loss functions

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


# objects to be used in the architectures

class ConvBlock(Layer):
    """TF Keras layer overriden to represent a convolutional block."""

    def __init__(self, filters=(64, ), kernel_size=(3, 3), activation='relu',
                 padding='same', dilation_rate=1, batch_norm=True,
                 dropout_rate=None, depth=2, strides=(1, 1),
                 kernel_initializer='glorot_uniform', name='conv_block',
                 **kwargs):
        """Create a block of two convolutional layers.

        Each of them could be followed by a batch normalization layer.

        :param filters: set of numbers of filters for each conv layer - if
            len(filters) == 1, the same number is used for every conv layer
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
        :param depth: depth of the block, specifying the number of conv
            layers in the block
        :param strides: An integer or tuple/list of 2 integers, specifying
            the strides of the convolution along the height and width
        :param kernel_initializer: initializer for the kernel weights matrix
        :param name: string base name of the layer
        :param kwargs: supplementary kwargs for the parent __init__()
        """
        super(ConvBlock, self).__init__(name=name, **kwargs)

        # set init parameters to member variables
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.depth = depth
        self.strides = strides
        self.kernel_initializer = kernel_initializer
        self.base_name = name
        self.kwargs = kwargs

        # solve the case of the same filter for each conv_layer
        if len(filters) == 1:
            filters = depth * filters

        # instantiate layers of the conv block
        self.conv_layers = []
        self.dropouts = []
        self.activations = []
        self.batch_norms = []
        for i in range(depth):
            self.conv_layers.append(
                Conv2D(filters[i], kernel_size, padding=padding,
                       dilation_rate=dilation_rate, strides=strides,
                       kernel_initializer=kernel_initializer,
                       name='{}_conv{}'.format(name, i)))
            if dropout_rate is not None:
                self.dropouts.append(Dropout(rate=dropout_rate))
            self.activations.append(Activation(activation))
            if self.batch_norm is True:
                self.batch_norms.append(BatchNormalization())

    def call(self, x, mask=None):
        """Perform the logic of applying the layer to the input tensors.

        :param x: input tensor
        :param mask: boolean tensor encoding masked timesteps in the input,
            used in RNN layers (currently not used)
        :return: output layer of the convolutional block
        """
        for i in range(self.depth):
            # apply inner blocks inside the entire block
            x = self.conv_layers[i](x)
            if self.dropout_rate is not None:
                x = self.dropouts[i](x)
            x = self.activations[i](x)
            if self.batch_norm is True:
                x = self.batch_norms[i](x)

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
                      depth=self.depth,
                      **self.kwargs)

        return config


class MyMaxPooling(Layer):
    """Custom implementation of a 2D max-pooling layer.

    Needed especially for SegNet to return also the pooling indices that are
    to be shared during the decoder phase.
    """

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None, **kwargs):
        """Construct the object and keep important variables.

        :param pool_size: Integer or tuple of 2 integers, window size over
            which to take the maximum
        :param strides: Integer, tuple of 2 integers, or None. Strides values
        :param padding: One of "valid" or 'same' (case-insensitive). 'valid'
            means no padding. "same" results in padding evenly distributed
        :param data_format: A string, one of 'channels_last' (default) or
            'channels_first'. The ordering of the dimensions in the inputs
            (so far not used)
        :param kwargs: TF Layer keyword arguments
        """
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

        super(MyMaxPooling, self).__init__(**kwargs)

    def call(self, x, mask=None):
        """Perform the logic of applying the layer to the input tensors.

        :param x: input tensor
        :param mask: boolean tensor encoding masked timesteps in the input,
            used in RNN layers (currently not used)
        :return: output layer of the convolutional block
        """
        ksize = (1, self.pool_size[0], self.pool_size[1], 1)
        strides = (1, self.strides[0], self.strides[1], 1)
        output, argmax = tf.nn.max_pool_with_argmax(
            x, ksize=ksize, strides=self.strides, padding=self.padding.upper(),
            include_batch_in_index=True)

        argmax = tf.cast(argmax, tf.int32)

        return output, argmax

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple (tuple of integers) or list of shape
            tuples (one per output tensor of the layer)
        :return: list describing the layer shape
        """
        ratio = (1, 2, 2, 1)
        output_shape = [dim // ratio[idx] if dim is not None else None
                        for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return output_shape, output_shape

    def compute_mask(self, inputs, mask=None):
        """Compute the output tensor mask.

        :param inputs: Tensor or list of tensors
        :param mask: Tensor or list of tensors
        :return: Tensor with the mask
        """
        return 2 * [None]

    def get_config(self):
        """Return the configuration of the convolutional block.

        Allows later reinstantiation (without its trained weights) from this
        configuration. It does not include connectivity information, nor the
        layer class name.

        :return: the configuration dictionary of the convolutional block
        """
        config = super(MyMaxPooling, self).get_config().copy()
        config.update({'pool_size': self.pool_size,
                       'padding': self.padding,
                       'strides': self.strides,
                       'data_format': self.data_format})

        return config


class MyMaxUnpooling(Layer):
    """Custom implementation of a 2D max-unpooling layer.

    Needed especially for SegNet to allow argmax-based unpooling with given
    indices.
    """

    def __init__(self, pool_size=(2, 2), data_format=None, **kwargs):
        """Construct the object and keep important variables.

        :param pool_size: Integer or tuple of 2 integers, window size over
            which to take the maximum
        :param data_format: A string, one of 'channels_last' (default) or
            'channels_first'. The ordering of the dimensions in the inputs
            (so far not used)
        :param kwargs: TF Layer keyword arguments
        """
        self.pool_size = pool_size

        # output shape should be created during the build() call
        self.output_shape_ = (None, None, None, None)

        super(MyMaxUnpooling, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        """Perform the logic of applying the layer to the input tensors.

        :param inputs: data structure in form (layer input, indices received
            from the corresponding max pooling layer)
        :param mask: boolean tensor encoding masked timesteps in the input,
            used in RNN layers (currently not used)
        :return: output layer of the upsampling block
        """
        x = inputs[0]
        indices = inputs[1]
        if indices is None:
            raise LayerDefinitionError('indices have to be specified')

        input_shape = tf.shape(x, out_type='int32')
        output_shape_complete = (input_shape[0],
                                 self.output_shape_[1],
                                 self.output_shape_[2],
                                 self.output_shape_[3])

        # unpool
        unpooled = tf.scatter_nd(keras.expand_dims(keras.flatten(indices)),
                                 keras.flatten(x),
                                 (keras.prod(output_shape_complete), ))

        # reshape
        unpooled = keras.reshape(unpooled, output_shape_complete)

        return unpooled

    def get_config(self):
        """Return the configuration of the convolutional block.

        Allows later reinstantiation (without its trained weights) from this
        configuration. It does not include connectivity information, nor the
        layer class name.

        :return: the configuration dictionary of the convolutional block
        """
        config = super(MyMaxUnpooling, self).get_config().copy()
        config.update({'pool_size': self.pool_size})

        return config

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple (tuple of integers) or list of shape
            tuples (one per output tensor of the layer)
        :return: list describing the layer shape
        """
        return (input_shape[0][0], self.output_shape_[1], self.output_shape_[2],
                self.output_shape_[3])

    def build(self, input_shape):
        """Create the input_shape class variable and build the layer.

        :param input_shape: Instance of TensorShape, or list of instances
            of TensorShape if the layer expects a list of inputs
        """
        self.output_shape_ = (input_shape[0][0],
                              input_shape[0][1] * self.pool_size[0],
                              input_shape[0][2] * self.pool_size[1],
                              input_shape[0][3])

        super(MyMaxUnpooling, self).build(input_shape)
