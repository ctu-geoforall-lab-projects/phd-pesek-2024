#!/usr/bin/python3

import os

import numpy as np
import tensorflow as tf

from osgeo import gdal
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers as k_layers
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, \
    Activation, Dropout, Add, AveragePooling2D, UpSampling2D, Concatenate
from tensorflow.keras import backend as keras

from data_preparation import generate_dataset_structure
from cnn_exceptions import LayerDefinitionError


class AugmentGenerator:
    """Data generator."""

    def __init__(self, data_dir, batch_size=5, operation='train',
                 tensor_shape=(256, 256), force_dataset_generation=False,
                 fit_memory=False, augment=False, onehot_encode=True,
                 val_set_pct=0.2, filter_by_class=None, verbose=1):
        """Initialize the generator.

        :param data_dir: path to the directory containing images
        :param batch_size: the number of samples that will be propagated
            through the network at once
        :param operation: either 'train' or 'val'
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
        :param verbose: verbosity (0=quiet, >0 verbose)
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
            generate_dataset_structure(data_dir, tensor_shape, val_set_pct,
                                       filter_by_class, augment,
                                       verbose=verbose)

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
        :return: yielded tuple of batch-sized np stacks of validation images
            and masks
        """
        # if self.augment is False:
        #     return self.generate_numpy(id2code)
        # else:
        #     return self.generate_augmented(id2code, seed)
        # TODO: Why does the TF approach not work?
        return self.generate_numpy(id2code)

    def generate_numpy(self, id2code):
        """Generate batches of data using our own numpy generator.

        Note: tf.data.Dataset.from_generator() seemed to be useful and maybe
        could speed up the process little bit , but it seemed not to work
        properly when __call__ takes arguments.

        :param id2code: dictionary mapping label ids to their codes
        :return: yielded tuple of batch-sized np stacks of validation images
            and masks
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
        :param batch_size: the number of samples that will be propagated
            through the network at once
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
        """Get a list of transposed images.

        :param data_dir: path to the directory containing images
        :param rescale: boolean saying whether to rescale images or not
            (rescaling is a division by 255)
        :return: list of transposed numpy matrices representing images in
            the dataset
        """
        # list of files from which the dataset will be created
        files_list = sorted(os.listdir(data_dir))

        images_list = [
            self.transpose_image(data_dir, file, rescale) for file in
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
            transposed *= 1. / 255

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
        encoded_image = np.empty(shape, dtype=np.float32)

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
    predictions = tf.cast(predictions, tf.float32, name='tversky_cast')
    ground_truth_onehot = tf.cast(ground_truth_onehot, tf.float32, name='tversky_cast_gt')

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

    def __init__(self, filters=(64, ), kernel_sizes=((3, 3), ),
                 activations=(k_layers.ReLU, ), paddings=('same', ),
                 dilation_rate=1, batch_norm=True, dropout_rate=None, depth=2,
                 strides=((1, 1), ), kernel_initializer='glorot_uniform',
                 use_bias=True, name='conv_block', **kwargs):
        """Create a block of convolutional layers.

        Each of them could be followed by a dropout layer, activation
        function, and/or batch normalization layer.

        :param filters: set of numbers of filters for each conv layer. If
            len(filters) == 1, the same number is used for every conv layer
        :param kernel_sizes: set of integers or tuples/lists of 2 integers,
            specifying the height and width of the 2D convolution window. If
            len(kernel_sizes) == 1, the same kernel is used for every conv
            layer
        :param activations: set of activation functions, such as tf.nn.relu,
            or string names of built-in activation function, such as 'relu'. If
            len(activations) == 1, the same activation function is used for
            every conv layer
        :param paddings: set of paddings for each conv layer. 'valid' means no
            padding. 'same' results in padding evenly to the left/right or
            up/down of the input such that output has the same height/width
            dimension as the input. If len(paddings) == 1, the same padding is
            used for every conv layer
        :param dilation_rate: convolution dilation rate
        :param batch_norm: boolean saying whether to use batch normalization
            or not
        :param dropout_rate: float between 0 and 1. Fraction of the input
            units of convolutional layers to drop
        :param depth: depth of the block, specifying the number of conv
            layers in the block
        :param strides: Set of integers or tuples/lists of 2 integers,
            specifying the strides of the convolution along the height and
            width. If len(strides) == 1, the same stride is used for every
            conv layer
        :param kernel_initializer: initializer for the kernel weights matrix
        :param use_bias: boolean saying whether the conv layers use a bias
            vector or not
        :param name: string base name of the block
        :param kwargs: supplementary kwargs for the parent __init__()
        """
        super(ConvBlock, self).__init__(name=name, **kwargs)

        # set init parameters to member variables
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.activations = activations
        self.paddings = paddings
        self.dilation_rate = dilation_rate
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.depth = depth
        self.strides = strides
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.base_name = name

        # solve the case of the same parameter for each conv_layer for the
        # variable ones
        if len(filters) == 1:
            self.filters = depth * filters
        if len(kernel_sizes) == 1:
            self.kernel_sizes = depth * kernel_sizes
        if len(activations) == 1:
            self.activations = depth * activations
        if len(paddings) == 1:
            self.paddings = depth * paddings
        if len(strides) == 1:
            self.strides = depth * strides

        # instantiate layers of the conv block
        self.conv_layers = []
        self.dropouts = []
        self.activation_layers = []
        self.batch_norms = []
        self.instantiate_layers()

    def call(self, inputs, mask=None, **kwargs):
        """Perform the logic of applying the layer to the input tensors.

        :param inputs: input tensor
        :param mask: boolean tensor encoding masked timesteps in the input,
            used in RNN layers (currently not used)
        :return: output layer of the convolutional block
        """
        x = inputs
        for i in range(self.depth):
            # apply inner blocks inside the entire block
            x = self.conv_layers[i](x)
            if self.dropout_rate is not None:
                x = self.dropouts[i](x)
            if self.activations[i] is not None:
                x = self.activation_layers[i](x)
            if self.batch_norm is True:
                x = self.batch_norms[i](x)

        return x

    def instantiate_layers(self):
        """Instantiate layers lying between the input and the classifier."""
        for i in range(self.depth):
            self.conv_layers.append(
                Conv2D(self.filters[i], self.kernel_sizes[i],
                       padding=self.paddings[i],
                       dilation_rate=self.dilation_rate,
                       strides=self.strides[i],
                       kernel_initializer=self.kernel_initializer,
                       use_bias=self.use_bias,
                       name='{}_conv{}'.format(self.base_name, i)))
            if self.dropout_rate is not None:
                self.dropouts.append(Dropout(rate=self.dropout_rate))
            if self.activations[i] is not None:
                self.activation_layers.append(self.activations[i]())
            if self.batch_norm is True:
                self.batch_norms.append(
                    BatchNormalization(
                        name='{}_bn{}'.format(self.base_name, i)))

    def get_config(self):
        """Return the configuration of the convolutional block.

        Allows later reinstantiation (without its trained weights) from this
        configuration. It does not include connectivity information, nor the
        layer class name.

        :return: the configuration dictionary of the convolutional block
        """
        config = super(ConvBlock, self).get_config()

        config.update(filters=self.filters,
                      kernel_sizes=self.kernel_sizes,
                      activations=self.activations,
                      paddings=self.paddings,
                      dilation_rate=self.dilation_rate,
                      batch_norm=self.batch_norm,
                      dropout_rate=self.dropout_rate,
                      depth=self.depth,
                      strides=self.strides,
                      kernel_initializer=self.kernel_initializer,
                      use_bias=self.use_bias)

        return config


class ResBlock(Layer):
    """TF Keras layer overriden to represent a residual block in ResNet.

    Following the definition of residual blocks for ResNet-50 and deeper from
    the original paper: <https://arxiv.org/pdf/1512.03385.pdf>. The original
    design was enhanced by the option to perform dropout.

    Represents only the better performing/more widely used version with 1x1
    shortcut convolution from the paper. The version with zero padding not
    implemented as I have never seen it anywhere in use.
    """

    def __init__(self, filters=(64, 64, 256), kernel_size=(3, 3),
                 activation=k_layers.ReLU, batch_norm=True, dropout_rate=None,
                 strides=(2, 2), use_bias=True, name='res_block', **kwargs):
        """Create a residual block.

        :param filters: set of numbers of filters for each conv layer
        :param kernel_size: an integer or tuple/list of 2 integers, specifying
            the height and width of the 2D convolution window in the central
            convolutional layer in the bottleneck block
        :param activation: activation function, such as tf.nn.relu, or string
            name of built-in activation function, such as 'relu'
        :param batch_norm: boolean saying whether to use batch normalization
            or not
        :param dropout_rate: float between 0 and 1. Fraction of the input
            units of convolutional layers to drop
        :param strides: integer or tuple/list of 2 integers, specifying
            the strides of the convolution along the height and width
        :param use_bias: boolean saying whether the conv layers use a bias
            vector or not
        :param name: string base name of the block
        :param kwargs: supplementary kwargs for the parent __init__()
        """
        super(ResBlock, self).__init__(name=name, **kwargs)

        # set init parameters to member variables
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.strides = strides
        self.use_bias = use_bias
        self.base_name = name

        # instantiate layers
        self.bottleneck = None
        self.shortcut = None
        self.add = None
        self.activation_layer = None
        self.instantiate_layers()

    def call(self, inputs, mask=None, **kwargs):
        """Perform the logic of applying the layer to the input tensors.

        :param inputs: Input tensor, or dict/list/tuple of input tensors
        :param mask: boolean tensor encoding masked timesteps in the input,
            used in RNN layers (currently not used)
        :return: output layer of the residual block
        """
        x = self.bottleneck(inputs)
        s = self.shortcut(inputs)
        x = self.add([x, s])
        x = self.activation_layer(x)

        return x

    def instantiate_layers(self):
        """Instantiate layers lying between the input and the output."""
        self.bottleneck = ConvBlock(filters=self.filters,
                                    kernel_sizes=((1, 1),
                                                  self.kernel_size,
                                                  (1, 1)),
                                    activations=(self.activation,
                                                 self.activation,
                                                 None),
                                    paddings=('valid',
                                              'same',
                                              'valid'),
                                    batch_norm=self.batch_norm,
                                    dropout_rate=self.dropout_rate,
                                    depth=3,
                                    strides=(self.strides,
                                             (1, 1),
                                             (1, 1)),
                                    use_bias=self.use_bias,
                                    kernel_initializer='he_normal',
                                    name=self.base_name + '_bottleneck')

        self.shortcut = ConvBlock(filters=(self.filters[-1], ),
                                  kernel_sizes=((1, 1), ),
                                  activations=(None, ),
                                  paddings=('valid', ),
                                  batch_norm=self.batch_norm,
                                  dropout_rate=self.dropout_rate,
                                  depth=1,
                                  strides=(self.strides, ),
                                  use_bias=self.use_bias,
                                  kernel_initializer='he_normal',
                                  name=self.base_name + '_shortcut')

        self.add = Add()
        self.activation_layer = self.activation()

    def get_config(self):
        """Return the configuration of the residual block.

        Allows later reinstantiation (without its trained weights) from this
        configuration. It does not include connectivity information, nor the
        layer class name.

        :return: the configuration dictionary of the residual block
        """
        config = super(ResBlock, self).get_config()

        config.update(filters=self.filters,
                      kernel_size=self.kernel_size,
                      activation=self.activation,
                      batch_norm=self.batch_norm,
                      dropout_rate=self.dropout_rate,
                      strides=self.strides,
                      use_bias=self.use_bias)

        return config


class IdentityBlock(Layer):
    """TF Keras layer overriden to represent an identity block in ResNet.

    Following the definition of residual blocks for ResNet-50 and deeper from
    the original paper: <https://arxiv.org/pdf/1512.03385.pdf>. The original
    design was enhanced by the option to perform dropout.
    """

    def __init__(self, filters=(64, 64, 256), kernel_size=(3, 3),
                 activation=k_layers.ReLU, batch_norm=True, dropout_rate=None,
                 strides=(2, 2), use_bias=True, name='res_block', **kwargs):
        """Create a residual block.

        :param filters: set of numbers of filters for each conv layer
        :param kernel_size: an integer or tuple/list of 2 integers, specifying
            the height and width of the 2D convolution window in the central
            convolutional layer in the bottleneck block
        :param activation: activation function, such as tf.nn.relu, or string
            name of built-in activation function, such as 'relu'
        :param batch_norm: boolean saying whether to use batch normalization
            or not
        :param dropout_rate: float between 0 and 1. Fraction of the input
            units of convolutional layers to drop
        :param strides: integer or tuple/list of 2 integers, specifying
            the strides of the convolution along the height and width
        :param use_bias: boolean saying whether the conv layers use a bias
            vector or not
        :param name: string base name of the block
        :param kwargs: supplementary kwargs for the parent __init__()
        """
        super(IdentityBlock, self).__init__(name=name, **kwargs)

        # set init parameters to member variables
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.strides = strides
        self.use_bias = use_bias
        self.base_name = name

        # instantiate layers
        self.bottleneck = None
        self.add = None
        self.activation_layer = None
        self.instantiate_layers()

    def call(self, inputs, mask=None, **kwargs):
        """Perform the logic of applying the layer to the input tensors.

        :param inputs: Input tensor, or dict/list/tuple of input tensors
        :param mask: boolean tensor encoding masked timesteps in the input,
            used in RNN layers (currently not used)
        :return: output layer of the residual block
        """
        x = self.bottleneck(inputs)
        x = self.add([x, inputs])
        x = self.activation_layer(x)

        return x

    def instantiate_layers(self):
        """Instantiate layers lying between the input and the output."""
        self.bottleneck = ConvBlock(filters=self.filters,
                                    kernel_sizes=((1, 1),
                                                  self.kernel_size,
                                                  (1, 1)),
                                    activations=(self.activation,
                                                 self.activation,
                                                 None),
                                    paddings=('valid',
                                              'same',
                                              'valid'),
                                    batch_norm=self.batch_norm,
                                    dropout_rate=self.dropout_rate,
                                    depth=3,
                                    strides=((1, 1),
                                             (1, 1),
                                             (1, 1)),
                                    use_bias=self.use_bias,
                                    kernel_initializer='he_normal',
                                    name=self.base_name + '_bottleneck')

        self.add = Add()
        self.activation_layer = self.activation()

    def get_config(self):
        """Return the configuration of the residual block.

        Allows later reinstantiation (without its trained weights) from this
        configuration. It does not include connectivity information, nor the
        layer class name.

        :return: the configuration dictionary of the residual block
        """
        config = super(IdentityBlock, self).get_config()

        config.update(filters=self.filters,
                      kernel_size=self.kernel_size,
                      activation=self.activation,
                      batch_norm=self.batch_norm,
                      dropout_rate=self.dropout_rate,
                      strides=self.strides,
                      use_bias=self.use_bias)

        return config


class ASPP(Layer):
    """TF Keras layer overriden to represent atrous spatial pyramid pooling.

    For the original paper, see <https://arxiv.org/pdf/1606.00915.pdf>.
    """

    def __init__(self, filters=256, kernel_size=(3, 3),
                 activation=k_layers.ReLU, batch_norm=True, dropout_rate=None,
                 dilation_rates=(1, 6, 12, 18, 24), pool_dims=(16, 16),
                 use_bias=True, name='aspp', **kwargs):
        """Create an atrous spatial pyramid pooling block.

        :param filters: number of filters for conv layers
        :param kernel_size: an integer or tuple/list of 2 integers, specifying
            the height and width of the 2D convolution window in the central
            convolutional layer in the bottleneck block
        :param activation: activation function, such as tf.nn.relu, or string
            name of built-in activation function, such as 'relu'
        :param batch_norm: boolean saying whether to use batch normalization
            or not
        :param dropout_rate: float between 0 and 1. Fraction of the input
            units of convolutional layers to drop
        :param dilation_rates: dilation rates used for convolutional blocks
            (the default values correspond to the original ASPP-L model)
        :param pool_dims: size of the pooling window for the pooling branch
            of the ASPP
        :param use_bias: boolean saying whether the conv layers use a bias
            vector or not
        :param name: string base name of the block
        :param kwargs: supplementary kwargs for the parent __init__()
        """
        super(ASPP, self).__init__(name=name, **kwargs)

        # set init parameters to member variables
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.dilation_rates = dilation_rates
        self.pool_dims = pool_dims
        self.use_bias = use_bias
        self.base_name = name

        # instantiate layers
        self.pool_blocks = None
        self.conv_blocks = []
        self.concat = None
        self.output_layer = None
        self.instantiate_layers()

    def call(self, inputs, mask=None, **kwargs):
        """Perform the logic of applying the layer to the input tensors.

        :param inputs: Input tensor, or dict/list/tuple of input tensors
        :param mask: boolean tensor encoding masked timesteps in the input,
            used in RNN layers (currently not used)
        :return: output layer of the residual block
        """
        x_pool = inputs
        for pool_block in self.pool_blocks:
            x_pool = pool_block(x_pool)

        block_outputs = [x_pool]
        for conv_block in self.conv_blocks:
            block_outputs.append(conv_block(inputs))

        # concat all outputs
        x = self.concat(block_outputs)

        # last (1, 1) convolution
        x = self.output_layer(x)

        return x

    def instantiate_layers(self):
        """Instantiate layers lying between the input and the output."""
        self.pool_blocks = [AveragePooling2D(pool_size=(self.pool_dims[0],
                                                        self.pool_dims[1]),
                                             name='average_pooling'),
                            ConvBlock(filters=(self.filters,),
                                      kernel_sizes=((1, 1),),
                                      activations=(self.activation, ),
                                      paddings=('same',),
                                      dilation_rate=1,
                                      batch_norm=self.batch_norm,
                                      dropout_rate=self.dropout_rate,
                                      depth=1,
                                      kernel_initializer='he_normal',
                                      use_bias=self.use_bias,
                                      name='ASPP_convblock_pool'),
                            UpSampling2D(size=[self.pool_dims[0] // 1,
                                               self.pool_dims[1] // 1],
                                         interpolation='bilinear')]

        for dilation_rate in self.dilation_rates:
            if dilation_rate == 1:
                kernel_size = (1, 1)
            else:
                kernel_size = self.kernel_size

            self.conv_blocks.append(
                ConvBlock(filters=(self.filters, ),
                          kernel_sizes=(kernel_size, ),
                          activations=(self.activation, ),
                          paddings=('same', ),
                          dilation_rate=dilation_rate,
                          batch_norm=self.batch_norm,
                          dropout_rate=self.dropout_rate,
                          depth=1,
                          kernel_initializer='he_normal',
                          use_bias=self.use_bias,
                          name=f'ASPP_convblock_d{dilation_rate}'))

        # concatenation layer
        self.concat = Concatenate(name='ASPP_concat')

        # output layer
        self.output_layer = ConvBlock(filters=(self.filters, ),
                                      kernel_sizes=(1, ),
                                      activations=(self.activation, ),
                                      paddings=('same', ),
                                      dilation_rate=1,
                                      dropout_rate=self.dropout_rate,
                                      depth=1,
                                      kernel_initializer='he_normal',
                                      use_bias=self.use_bias,
                                      name=f'ASPP_convblock_final')

    def get_config(self):
        """Return the configuration of the residual block.

        Allows later reinstantiation (without its trained weights) from this
        configuration. It does not include connectivity information, nor the
        layer class name.

        :return: the configuration dictionary of the residual block
        """
        config = super(ASPP, self).get_config()

        config.update(filters=self.filters,
                      kernel_size=self.kernel_size,
                      activation=self.activation,
                      batch_norm=self.batch_norm,
                      dropout_rate=self.dropout_rate,
                      dilation_rates=self.dilation_rates,
                      pool_dims=self.pool_dims,
                      use_bias=self.use_bias)

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
        super(MyMaxPooling, self).__init__(**kwargs)

        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format

        # TODO: self.instantiate_layers()

    def call(self, inputs, mask=None, **kwargs):
        """Perform the logic of applying the layer to the input tensors.

        :param inputs: input tensor
        :param mask: boolean tensor encoding masked timesteps in the input,
            used in RNN layers (currently not used)
        :return: output layer of the convolutional block
        """
        ksize = (1, self.pool_size[0], self.pool_size[1], 1)
        # TODO: Why don't I use the following strides?
        strides = (1, self.strides[0], self.strides[1], 1)
        output, argmax = tf.nn.max_pool_with_argmax(
            inputs, ksize=ksize, strides=self.strides,
            padding=self.padding.upper(), include_batch_in_index=True)

        # argmax = tf.cast(argmax, tf.int32, name='cast_maxpooling')
        argmax = tf.cast(argmax, tf.float32, name='cast_maxpooling')

        return output, argmax

    @staticmethod
    def compute_output_shape(input_shape, **kwargs):
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

    @staticmethod
    def compute_mask(inputs, mask=None, **kwargs):
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
        config = super(MyMaxPooling, self).get_config()

        config.update(pool_size=self.pool_size,
                      padding=self.padding,
                      strides=self.strides,
                      data_format=self.data_format)

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
        super(MyMaxUnpooling, self).__init__(**kwargs)

        self.pool_size = pool_size
        self.data_format = data_format

        # output shape should be created during the build() call
        self.output_shape_ = (None, None, None, None)

        # TODO: self.instantiate_layers()

    def call(self, inputs, mask=None, **kwargs):
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
        config = super(MyMaxUnpooling, self).get_config()

        config.update(pool_size=self.pool_size,
                      data_format=self.data_format)

        return config

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple (tuple of integers) or list of shape
            tuples (one per output tensor of the layer)
        :return: list describing the layer shape
        """
        return (input_shape[0][0], self.output_shape_[1],
                self.output_shape_[2], self.output_shape_[3])

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
