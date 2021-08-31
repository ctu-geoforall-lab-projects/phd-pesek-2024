#!/usr/bin/python3

import tensorflow as tf

from abc import ABC, abstractmethod
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Input, UpSampling2D, \
    Concatenate, Dropout
from tensorflow.keras.models import Model

from cnn_lib import ConvBlock, MyMaxPooling, MyMaxUnpooling
from cnn_exceptions import ModelConfigError


class _BaseModel(Model, ABC):
    """A base Model class holding methods mutual for various architectures."""

    def __init__(self, nr_classes, nr_bands=12, nr_filters=64, batch_norm=True,
                 dilation_rate=1, tensor_shape=(256, 256), activation='relu',
                 padding='same', dropout_rate_input=None,
                 dropout_rate_hidden=None):
        """Model constructor.

        :param nr_classes: number of classes to be predicted
        :param nr_bands: number of bands of intended input images
        :param nr_filters: base number of convolution filters (multiplied
            deeper in the model)
        :param batch_norm: boolean saying whether to use batch normalization
            or not
        :param dilation_rate: convolution dilation rate
        :param tensor_shape: shape of the first two dimensions of input tensors
        :param activation: activation function, such as tf.nn.relu, or string
            name of built-in activation function, such as 'relu'
        :param padding: 'valid' means no padding. 'same' results in padding
            evenly to the left/right or up/down of the input such that output
            has the same height/width dimension as the input
        :param dropout_rate_input: float between 0 and 1. Fraction of the input
            units of the input layer to drop
        :param dropout_rate_hidden: float between 0 and 1. Fraction of
            the input
        """
        super(_BaseModel, self).__init__()

        self.nr_classes = nr_classes
        self.nr_bands = nr_bands
        self.nr_filters = nr_filters
        self.batch_norm = batch_norm
        self.dilation_rate = dilation_rate
        self.tensor_shape = tensor_shape
        self.activation = activation
        self.padding = padding
        self.dropout_rate_input = dropout_rate_input
        self.dropout_rate_hidden = dropout_rate_hidden

        self.check_parameters()

        # layers instantiation
        self.dropout_in = self.get_input_dropout_layer()
        # a call to self.instantiate_layers() returning the rest should be here
        # for children classes
        self.outputs = self.get_classifier_layer()

    def check_parameters(self):
        """Check the reasonability of the architecture parameters."""
        if any([i % (2 ** 4) != 0 for i in self.tensor_shape]):
            raise ModelConfigError(
                'The tensor height and tensor width must be devidable by 32 '
                'for the architecture, but they are {} and {} '
                'respectively instead'.format(self.tensor_shape[0],
                                              self.tensor_shape[1])
            )

    def get_classifier_function(self):
        """Choose the activation function for the last layer.

        :return: string containing the name of the activation function
        """
        if self.nr_classes == 2:
            classifier_activation = 'sigmoid'
        else:
            classifier_activation = 'softmax'

        return classifier_activation

    def get_input_dropout_layer(self):
        """Apply dropout to the input layer if wanted.

        :return: dropout layer
        """
        if self.dropout_rate_input is not None:
            x = Dropout(rate=self.dropout_rate_input, name='dropout_input')
        else:
            x = lambda a: a

        return x

    def get_classifier_layer(self):
        """Get the classifier layer.

        :return: the classifier layer
        """
        return Conv2D(self.nr_classes,
                      (1, 1),
                      activation=self.get_classifier_function(),
                      padding=self.padding,
                      dilation_rate=self.dilation_rate,
                      name='classifier_layer')

    def summary(self, line_length=None, positions=None, print_fn=None):
        """Print a string summary of the network.

        Must be overriden with the Input layer defined because of a bug in
        TF. However, this solution also prints the input layer - that one is
        not actually part of the network.

        :param line_length: Total length of printed lines
        :param positions: Relative or absolute positions of log elements
            in each line
        :param print_fn: Print function to use
        :return: printed string summary of the network
        """
        inputs = Input((self.tensor_shape[0], self.tensor_shape[1],
                        self.nr_bands))
        model = Model(inputs=[inputs],
                      outputs=self.call(inputs))
        return model.summary()

    @abstractmethod
    def instantiate_layers(self):
        """Instantiate layers lying between the input and the classifier.

        TODO: Maybe the layers could be put defined as class variables instead
              of returned values?

        :return: this thing unfortunately differs
        """
        pass


class UNet(_BaseModel):
    """U-Net architecture.

    For the original paper, see <https://arxiv.org/pdf/1505.04597.pdf>.
    The original architecture was enhanced by the option to perform dropout
    and batch normalization and to specify padding (no padding in the
    original - cropping would be needed in such case).
    """

    def __init__(self, *args, **kwargs):
        """Model constructor.

        :param nr_classes: number of classes to be predicted
        :param nr_bands: number of bands of intended input images
        :param nr_filters: base number of convolution filters (multiplied
            deeper in the model)
        :param batch_norm: boolean saying whether to use batch normalization
            or not
        :param dilation_rate: convolution dilation rate
        :param tensor_shape: shape of the first two dimensions of input tensors
        :param activation: activation function, such as tf.nn.relu, or string
            name of built-in activation function, such as 'relu'
        :param padding: 'valid' means no padding. 'same' results in padding
            evenly to the left/right or up/down of the input such that output
            has the same height/width dimension as the input
        :param dropout_rate_input: float between 0 and 1. Fraction of the input
            units of the input layer to drop
        :param dropout_rate_hidden: float between 0 and 1. Fraction of
            the input
        """
        super(UNet, self).__init__(*args, **kwargs)

        ds_layers, self.m_block, us_layers = self.instantiate_layers()
        self.ds_blocks = ds_layers[0]
        self.ds_pools = ds_layers[1]
        self.us_pools = us_layers[0]
        self.us_convs = us_layers[1]
        self.us_concats = us_layers[2]
        self.us_blocks = us_layers[3]

    def call(self, inputs, training=None, mask=None):
        """Call the model on new inputs.

        :param inputs: Input tensor, or dict/list/tuple of input tensors
        :param training: Boolean or boolean scalar tensor, indicating whether
            to run the Network in training mode or inference mode
        :param mask: A mask or list of masks
        :return: the output of the classifier layer
        """
        x = self.dropout_in(tf.cast(inputs, tf.float16))

        # downsampling
        x, concat_layers = self.run_downsampling_section(x)

        # middle block
        x = self.m_block(x)

        # upsampling
        x = self.run_upsampling_section(x, concat_layers)

        # softmax classifier head layer
        classes = self.outputs(x)

        return classes

    def instantiate_layers(self):
        """Instantiate layers lying between the input and the classifier.

        TODO: Maybe the layers could be put defined as class variables instead
              of returned values?

        :return: this thing unfortunately differs
        """
        # downsampling layers
        ds_blocks = []
        ds_pools = []
        for i in range(4):
            ds_blocks.append(ConvBlock(self.nr_filters * (2 ** i),
                                       (3, 3),
                                       self.activation,
                                       self.padding,
                                       self.dilation_rate,
                                       dropout_rate=self.dropout_rate_hidden,
                                       depth=2,
                                       name=f'downsampling_block{i}'))
            ds_pools.append(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                                         data_format='channels_last',
                                         name=f'downsampling_pooling{i}'))

        ds_ret = (ds_blocks, ds_pools)

        # middle conv block
        m_block = ConvBlock(self.nr_filters * (2 ** 4), (3, 3),
                            self.activation, self.padding, self.dilation_rate,
                            dropout_rate=self.dropout_rate_hidden, depth=2,
                            name='middle_block')

        # upsampling layers
        us_samples = []
        us_convs = []
        us_concats = []
        us_blocks = []
        for i in range(3, -1, -1):
            us_samples.append(UpSampling2D(size=(2, 2),
                                           name=f'upsampling_pool{i}'))
            us_convs.append(Conv2D(self.nr_filters * (2 ** i), (2, 2),
                                   padding=self.padding,
                                   dilation_rate=self.dilation_rate,
                                   name=f'upsampling_conv{i}'))
            # concatenate the upsampled weights with the corresponding ones
            # from the contracting path
            us_concats.append(Concatenate(axis=3, name=f'upsampling_concat{i}'))
            us_blocks.append(ConvBlock(self.nr_filters * (2 ** i),
                                       (3, 3),
                                       self.activation,
                                       self.padding,
                                       self.dilation_rate,
                                       dropout_rate=self.dropout_rate_hidden,
                                       depth=2,
                                       name=f'upsampling_block{i}'))

        us_ret = (us_samples, us_convs, us_concats, us_blocks)

        return ds_ret, m_block, us_ret

    def run_downsampling_section(self, x):
        """Run U-Net downsampling.

        :param x: input tensor
        :return: output of the downsampling, list of layer outputs to be
            concatenated
        """
        concat_layers = []

        for i in range(4):
            x = self.ds_blocks[i](x)
            concat_layers.append(x)
            x = self.ds_pools[i](x)

        return x, concat_layers

    def run_upsampling_section(self, x, concat_layers):
        """Run U-Net upsampling.

        :param x: input tensor
        :param concat_layers: list of layer outputs from downsampling to be
            concatenated
        :return: output of the upsampling
        """
        for i in range(4):
            x = self.us_pools[i](x)
            x = self.us_convs[i](x)
            # concatenate the upsampled weights with the corresponding ones
            # from the contracting path
            x = self.us_concats[i]([x, concat_layers[-(i + 1)]])
            x = self.us_blocks[i](x)

        return x


class SegNet(_BaseModel):
    """SegNet architecture.

    For the original paper, see <https://arxiv.org/pdf/1511.00561.pdf>.
    The original architecture was enhanced by the option to perform dropout
    and batch normalization and to specify padding.
    """

    def __init__(self, *args, **kwargs):
        """Model constructor.

        :param nr_classes: number of classes to be predicted
        :param nr_bands: number of bands of intended input images
        :param nr_filters: base number of convolution filters (multiplied
            deeper in the model)
        :param batch_norm: boolean saying whether to use batch normalization
            or not
        :param dilation_rate: convolution dilation rate
        :param tensor_shape: shape of the first two dimensions of input tensors
        :param activation: activation function, such as tf.nn.relu, or string
            name of built-in activation function, such as 'relu'
        :param padding: 'valid' means no padding. 'same' results in padding
            evenly to the left/right or up/down of the input such that output
            has the same height/width dimension as the input
        :param dropout_rate_input: float between 0 and 1. Fraction of the input
            units of the input layer to drop
        :param dropout_rate_hidden: float between 0 and 1. Fraction of
            the input
        """
        super(SegNet, self).__init__(*args, **kwargs)

        ds_layers, us_layers = self.instantiate_layers()
        self.ds_blocks = ds_layers[0]
        self.ds_pools = ds_layers[1]
        self.us_pools = us_layers[0]
        self.us_blocks = us_layers[1]

    def call(self, inputs, training=None, mask=None):
        """Call the model on new inputs.

        :param inputs: Input tensor, or dict/list/tuple of input tensors
        :param training: Boolean or boolean scalar tensor, indicating whether
            to run the Network in training mode or inference mode
        :param mask: A mask or list of masks
        :return: the output of the classifier layer
        """
        x = self.dropout_in(tf.cast(inputs, tf.float16))

        # downsampling
        x, pool_indices = self.run_downsampling_section(x)

        # upsampling
        x = self.run_upsampling_section(x, pool_indices)

        # softmax classifier head layer
        classes = self.outputs(x)

        return classes

    def instantiate_layers(self):
        """Instantiate layers lying between the input and the classifier.

        TODO: Maybe the layers could be put defined as class variables instead
              of returned values?

        :return: this thing unfortunately differs
        """
        # downsampling layers
        ds_blocks = []
        ds_pools = []
        for i in range(2):
            # blocks of the depth 2
            ds_blocks.append(ConvBlock(self.nr_filters * (2 ** i),
                                       (3, 3),
                                       self.activation, self.padding,
                                       self.dilation_rate,
                                       dropout_rate=self.dropout_rate_hidden,
                                       depth=2))
            ds_pools.append(MyMaxPooling(pool_size=(2, 2),
                                         strides=(2, 2),
                                         data_format='channels_last'))

        for i in range(2, 5):
            # blocks of the depth 3
            ds_blocks.append(ConvBlock(self.nr_filters * (2 ** i),
                                       (3, 3),
                                       self.activation,
                                       self.padding,
                                       self.dilation_rate,
                                       dropout_rate=self.dropout_rate_hidden,
                                       depth=3))
            ds_pools.append(MyMaxPooling(pool_size=(2, 2),
                                         strides=(2, 2),
                                         data_format='channels_last'))

        ds_ret = (ds_blocks, ds_pools)

        # upsampling layers
        us_samples = []
        us_blocks = []
        for i in range(4, 1, -1):
            # blocks of the depth 3
            # upsampling with shared indices
            us_samples.append(MyMaxUnpooling(pool_size=(2, 2)))
            us_blocks.append(ConvBlock(self.nr_filters * (2 ** i),
                                       (3, 3),
                                       self.activation,
                                       self.padding,
                                       self.dilation_rate,
                                       dropout_rate=self.dropout_rate_hidden,
                                       depth=2))
            us_blocks.append(ConvBlock(self.nr_filters * (2 ** (i - 1)),
                                       (3, 3),
                                       self.activation,
                                       self.padding,
                                       self.dilation_rate,
                                       dropout_rate=self.dropout_rate_hidden,
                                       depth=1))

        # a block of the depth 2
        us_samples.append(MyMaxUnpooling(pool_size=(2, 2)))
        us_blocks.append(ConvBlock(self.nr_filters * (2 ** 1),
                                   (3, 3),
                                   self.activation,
                                   self.padding,
                                   self.dilation_rate,
                                   dropout_rate=self.dropout_rate_hidden,
                                   depth=1))
        us_blocks.append(ConvBlock(self.nr_filters * (2 ** 0),
                                   (3, 3),
                                   self.activation,
                                   self.padding,
                                   self.dilation_rate,
                                   dropout_rate=self.dropout_rate_hidden,
                                   depth=1))

        # a block of the depth 1
        # the paper states depth two and then softmax, but I believe that this
        # should do the same trick
        us_samples.append(MyMaxUnpooling(pool_size=(2, 2)))
        us_blocks.append(ConvBlock(self.nr_filters * (2 ** 0),
                                   (3, 3),
                                   self.activation,
                                   self.padding,
                                   self.dilation_rate,
                                   dropout_rate=self.dropout_rate_hidden,
                                   depth=1))

        us_ret = (us_samples, us_blocks)

        return ds_ret, us_ret

    def run_downsampling_section(self, x):
        """Run SegNet downsampling.

        :param x: input tensor
        :return: output of the downsampling, list of layer outputs to be
            concatenated
        """
        pool_indices = []

        for i in range(len(self.ds_blocks)):
            x = self.ds_blocks[i](x)
            x, pi = self.ds_pools[i](x)
            pool_indices.append(pi)

        return x, pool_indices

    def run_upsampling_section(self, x, pool_indices):
        """Run SegNet upsampling.

        :param x: input tensor
        :param pool_indices: indices from the downsampling pooling layers to
            be used for the upsampling
        :return: output of the upsampling
        """
        for i in range(len(self.us_pools)):
            x = self.us_pools[i]((x, pool_indices[- (i + 1)]))
            x = self.us_blocks[2 * i](x)
            if 2 * i + 1 < len(self.us_blocks):
                x = self.us_blocks[2 * i + 1](x)

        return x
