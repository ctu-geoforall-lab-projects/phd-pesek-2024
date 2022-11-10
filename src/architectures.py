#!/usr/bin/python3

import tensorflow as tf

from abc import ABC, abstractmethod
from tensorflow.keras import layers as k_layers
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Input, \
    UpSampling2D, Concatenate, Dropout, ZeroPadding2D, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Dense
from tensorflow.keras.models import Model

from cnn_lib import ConvBlock, MyMaxPooling, MyMaxUnpooling, \
    categorical_dice, categorical_tversky, ResBlock, IdentityBlock, ASPP
from cnn_exceptions import ModelConfigError


class _BaseModel(Model, ABC):
    """A base Model class holding methods mutual for various architectures."""

    def __init__(self, nr_classes, nr_bands=12, nr_filters=64, batch_norm=True,
                 dilation_rate=1, tensor_shape=(256, 256),
                 activation=k_layers.ReLU,
                 padding='same', dropout_rate_input=None,
                 dropout_rate_hidden=None, use_bias=True, name='model', **kwargs):
        """Model constructor.

        :param nr_classes: number of classes to be predicted
        :param nr_bands: number of bands of intended input images
        :param nr_filters: base number of convolution filters (multiplied
            deeper in the model)
        :param batch_norm: boolean saying whether to use batch normalization
            or not
        :param dilation_rate: convolution dilation rate
        :param tensor_shape: shape of the first two dimensions of input tensors
        :param activation: activation layer
        :param padding: 'valid' means no padding. 'same' results in padding
            evenly to the left/right or up/down of the input such that output
            has the same height/width dimension as the input
        :param dropout_rate_input: float between 0 and 1. Fraction of the input
            units of the input layer to drop
        :param dropout_rate_hidden: float between 0 and 1. Fraction of
            the input
        :param name: The name of the model
        :param use_bias: Boolean, whether the layer uses a bias vector
        """
        super(_BaseModel, self).__init__(name=name, **kwargs)

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
        self.use_bias = use_bias
        # TODO: Maybe use_bias should be by default == False, see:
        #       https://arxiv.org/pdf/1502.03167.pdf

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
                'The tensor height and tensor width must be divisible by 32 '
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
                        self.nr_bands), dtype=tf.float16, name='input')
        model = Model(inputs=[inputs], outputs=self.call(inputs),
                      name=self.name)
        return model.summary()

    @abstractmethod
    def instantiate_layers(self):
        """Instantiate layers lying between the input and the classifier.

        TODO: Maybe the layers could be put defined as class variables instead
              of returned values?

        :return: this thing unfortunately differs
        """
        pass

    def get_config(self):
        """Return the configuration of the convolutional block.

        Allows later reinstantiation (without its trained weights) from this
        configuration. It does not include connectivity information, nor the
        model class name.

        :return: the configuration dictionary of the convolutional block
        """
        return super(_BaseModel, self).get_config()


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
        x = self.dropout_in(tf.cast(inputs, tf.float16, name='type_cast'))

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
            ds_blocks.append(ConvBlock((self.nr_filters * (2 ** i), ),
                                       ((3, 3), ),
                                       (self.activation, ),
                                       (self.padding, ),
                                       self.dilation_rate,
                                       dropout_rate=self.dropout_rate_hidden,
                                       depth=2,
                                       name=f'downsampling_block{i}'))
            ds_pools.append(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                                         data_format='channels_last',
                                         name=f'downsampling_pooling{i}'))

        ds_ret = (ds_blocks, ds_pools)

        # middle conv block
        m_block = ConvBlock((self.nr_filters * (2 ** 4), ), ((3, 3), ),
                            (self.activation, ), (self.padding, ),
                            self.dilation_rate,
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
            us_concats.append(Concatenate(axis=3,
                                          name=f'upsampling_concat{i}'))
            us_blocks.append(ConvBlock((self.nr_filters * (2 ** i), ),
                                       ((3, 3), ),
                                       (self.activation, ),
                                       (self.padding, ),
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
        x = self.dropout_in(tf.cast(inputs, tf.float16, name='type_cast'))

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
            ds_blocks.append(ConvBlock((self.nr_filters * (2 ** i), ),
                                       ((3, 3), ),
                                       (self.activation, ), (self.padding, ),
                                       self.dilation_rate,
                                       dropout_rate=self.dropout_rate_hidden,
                                       depth=2,
                                       name=f'downsampling_block{i}'))
            ds_pools.append(MyMaxPooling(pool_size=(2, 2),
                                         strides=(2, 2),
                                         data_format='channels_last'))

        for i in range(2, 5):
            # blocks of the depth 3
            ds_blocks.append(ConvBlock((self.nr_filters * (2 ** i), ),
                                       ((3, 3), ),
                                       (self.activation, ),
                                       (self.padding, ),
                                       self.dilation_rate,
                                       dropout_rate=self.dropout_rate_hidden,
                                       depth=3,
                                       name=f'downsampling_block{i}'))
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
            us_blocks.append(ConvBlock((self.nr_filters * (2 ** i), ),
                                       ((3, 3), ),
                                       (self.activation, ),
                                       (self.padding, ),
                                       self.dilation_rate,
                                       dropout_rate=self.dropout_rate_hidden,
                                       depth=2,
                                       name=f'upsampling_block{i}_2'))
            us_blocks.append(ConvBlock((self.nr_filters * (2 ** (i - 1)), ),
                                       ((3, 3), ),
                                       (self.activation, ),
                                       (self.padding, ),
                                       self.dilation_rate,
                                       dropout_rate=self.dropout_rate_hidden,
                                       depth=1,
                                       name=f'upsampling_block{i}_1'))

        # a block of the depth 2
        us_samples.append(MyMaxUnpooling(pool_size=(2, 2)))
        us_blocks.append(ConvBlock((self.nr_filters * (2 ** 1), ),
                                   ((3, 3), ),
                                   (self.activation, ),
                                   (self.padding, ),
                                   self.dilation_rate,
                                   dropout_rate=self.dropout_rate_hidden,
                                   depth=1,
                                   name=f'upsampling_block1_2'))
        us_blocks.append(ConvBlock((self.nr_filters * (2 ** 0), ),
                                   ((3, 3), ),
                                   (self.activation, ),
                                   (self.padding, ),
                                   self.dilation_rate,
                                   dropout_rate=self.dropout_rate_hidden,
                                   depth=1,
                                   name=f'upsampling_block1_1'))

        # a block of the depth 1
        # the paper states depth two and then softmax, but I believe that this
        # should do the same trick
        us_samples.append(MyMaxUnpooling(pool_size=(2, 2)))
        us_blocks.append(ConvBlock((self.nr_filters * (2 ** 0), ),
                                   ((3, 3), ),
                                   (self.activation, ),
                                   (self.padding, ),
                                   self.dilation_rate,
                                   dropout_rate=self.dropout_rate_hidden,
                                   depth=1,
                                   name=f'upsampling_block0'))

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


class ResNet(_BaseModel):
    """ResNet architecture.

    For the original paper, see <https://arxiv.org/pdf/1512.03385.pdf>.
    The original architecture was enhanced by the option to perform dropout.
    Another change is the fact that the batch normalization is used after
    activation functions, not before them - to see motivation for this step,
    see the following links:
    <https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/>
    <https://blog.paperspace.com/busting-the-myths-about-batch-normalization/>
    <https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout>
    """

    def __init__(self, *args, pooling='avg', depth=50, include_top=True,
                 return_layers=None, **kwargs):
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
        :param pooling: global pooling mode for feature extraction
            (must be 'avg' or 'max')
        :param depth: depth of the ResNet model (must be 50, 101, or 152)
        :param include_top: whether to include the fully-connected layer
            at the top of the network
        :param return_layers: layers to be returned (allows multistage
            returns for the usage of ResNet as a backbone architecture)
        """
        if pooling not in ('avg', 'max'):
            raise ModelConfigError(
                f'Pooling {pooling} not supported for ResNet. Supported '
                f'pooling values are "avg" and "max"')
        if depth not in (50, 101, 152):
            raise ModelConfigError(
                f'ResNet variant of depth {depth} not supported. Supported '
                f'depths are 50, 101, and 152')

        self.pooling = pooling
        self.depth = depth
        self.include_top = include_top
        self.return_layers = return_layers

        super(ResNet, self).__init__(*args, **kwargs)

        # get depths of individual ResNet stages depending on total depth
        self.depths = self.get_stage_depths(depth)
        self.resnet_layers = self.instantiate_layers()

    def call(self, inputs, training=None, mask=None):
        """Call the model on new inputs.

        :param inputs: Input tensor, or dict/list/tuple of input tensors
        :param training: Boolean or boolean scalar tensor, indicating whether
            to run the Network in training mode or inference mode
        :param mask: A mask or list of masks
        :return: the output of the last layer
            (either classifier or pooling for the case of the backbone usage)
        """
        x = self.dropout_in(tf.cast(inputs, tf.float16, name='type_cast'))

        # run resnet
        return_outputs = []  # used if self.return_layers is not None
        for layer in self.resnet_layers:
            x = layer(x)

            if self.return_layers is not None:
                if layer.name in self.return_layers:
                    return_outputs.append(x)
                    if len(return_outputs) == len(self.return_layers):
                        return return_outputs

        # TODO: Situation with return layers and self.include_top is True
        # classifier head layer
        if self.outputs is not None:
            x = self.outputs(x)

        return x

    def get_config(self):
        """Return the configuration of the convolutional block.

        Allows later reinstantiation (without its trained weights) from this
        configuration. It does not include connectivity information, nor the
        model class name.

        :return: the configuration dictionary of the convolutional block
        """
        config = super(ResNet, self).get_config()

        config.update(pooling=self.pooling,
                      depth=self.depth,
                      include_top=self.include_top,
                      return_layers=self.return_layers)

        return config

    def instantiate_layers(self):
        """Instantiate layers lying between the input and the classifier.

        TODO: Maybe the layers could be put defined as class variables instead
              of returned values?

        :return: this thing unfortunately differs
        """
        # stage 1
        stage1 = [ZeroPadding2D(padding=(3, 3), name='conv1_pad'),
                  ConvBlock(filters=(64,),
                            kernel_sizes=((7, 7),),
                            activations=(self.activation,),
                            paddings=('valid',),
                            dropout_rate=self.dropout_rate_hidden,
                            depth=1,
                            strides=((2, 2),),
                            use_bias=self.use_bias,
                            kernel_initializer='he_normal',
                            name='conv_block_1'),
                  ZeroPadding2D(padding=(1, 1), name='pool1_pad'),
                  MaxPooling2D((3, 3), strides=(2, 2))]
        # TODO: Why zero padding?

        # stage 2
        stage2 = [ResBlock(kernel_size=3,
                           filters=(64, 64, 256),
                           dropout_rate=self.dropout_rate_hidden,
                           strides=(1, 1),
                           activation=self.activation,
                           use_bias=self.use_bias,
                           name='res_block_2_1')]
        for i in range(2, self.depths[1] + 1):
            stage2.append(IdentityBlock(kernel_size=3,
                                        filters=(64, 64, 256),
                                        activation=self.activation,
                                        dropout_rate=self.dropout_rate_hidden,
                                        use_bias=self.use_bias,
                                        name=f'id_block_2_{i}'))

        # stage 3
        stage3 = [ResBlock(kernel_size=3,
                           filters=(128, 128, 512),
                           activation=self.activation,
                           dropout_rate=self.dropout_rate_hidden,
                           use_bias=self.use_bias,
                           name='res_block_3_1')]
        for i in range(2, self.depths[2] + 1):
            stage3.append(IdentityBlock(kernel_size=3,
                                        filters=(128, 128, 512),
                                        activation=self.activation,
                                        dropout_rate=self.dropout_rate_hidden,
                                        use_bias=self.use_bias,
                                        name=f'id_block_3_{i}'))

        # stage 4
        stage4 = [ResBlock(kernel_size=3,
                           filters=(256, 256, 1024),
                           use_bias=self.use_bias,
                           activation=self.activation,
                           dropout_rate=self.dropout_rate_hidden,
                           name='res_block_4_1')]
        for i in range(2, self.depths[3] + 1):
            stage4.append(IdentityBlock(kernel_size=3,
                                        filters=(256, 256, 1024),
                                        use_bias=self.use_bias,
                                        activation=self.activation,
                                        dropout_rate=self.dropout_rate_hidden,
                                        name=f'id_block_4_{i}'))

        # stage 5
        stage5 = [ResBlock(kernel_size=3,
                           filters=(512, 512, 2048),
                           use_bias=self.use_bias,
                           activation=self.activation,
                           dropout_rate=self.dropout_rate_hidden,
                           name='res_block_5_1')]
        for i in range(2, self.depths[4] + 1):
            stage5.append(IdentityBlock(kernel_size=3,
                                        filters=(512, 512, 2048),
                                        use_bias=self.use_bias,
                                        activation=self.activation,
                                        dropout_rate=self.dropout_rate_hidden,
                                        name=f'id_block_5_{i}'))

        # top
        if self.pooling == 'avg':
            top = [GlobalAveragePooling2D()]
        else:
            # self.pooling == 'max'
            top = [GlobalMaxPooling2D()]

        return stage1 + stage2 + stage3 + stage4 + stage5 + top

    def get_classifier_layer(self):
        """Get the classifier layer.

        :return: the classifier layer
        """
        if self.include_top is True:
            return Dense(self.nr_classes, activation=self.activation,
                         name='classifier_layer')
        else:
            return None

    @staticmethod
    def get_stage_depths(depth):
        """Get depths corresponding to individual stages of ResNet.

        :param depth: depth of the ResNet model
        :return: a tuple of depths corresponding to individual stages of ResNet
        """
        stage_2_depth = 3
        if depth == 50:
            stage_3_depth = 4
            stage_4_depth = 6
        elif depth == 101:
            stage_3_depth = 4
            stage_4_depth = 23
        else:
            # depth == 152
            stage_3_depth = 8
            stage_4_depth = 36
        stage_5_depth = 3

        return 1, stage_2_depth, stage_3_depth, stage_4_depth, stage_5_depth


class DeepLabv3Plus(_BaseModel):
    """DeeLabv3+ architecture.

    For the original paper, see <https://arxiv.org/pdf/1802.02611.pdf>.
    The original architecture was enhanced by the option to perform dropout.
    """

    def __init__(self, *args, resnet_pooling='avg', resnet_depth=50,
                 resnet_2_out='id_block_4_6', **kwargs):
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
        :param resnet_pooling: global pooling mode for feature extraction
            in the backbone ResNet model (must be 'avg' or 'max')
        :param resnet_2_out: ResNet layer to be passed into ASPP
        :param resnet_depth: depth of the ResNet backbone model
            (must be 50, 101, or 152)
        """
        self.resnet_pooling = resnet_pooling
        self.resnet_depth = resnet_depth
        self.resnet_2_out = resnet_2_out

        super(DeepLabv3Plus, self).__init__(*args, **kwargs)

        # instantiate layers
        self.backbone = None
        self.aspp = None
        self.aspp_upsample = None
        self.low_level = None
        self.concat = None
        self.decoder_layers = None
        self.instantiate_layers()

    def call(self, inputs, training=None, mask=None):
        """Call the model on new inputs.

        :param inputs: Input tensor, or dict/list/tuple of input tensors
        :param training: Boolean or boolean scalar tensor, indicating whether
            to run the Network in training mode or inference mode
        :param mask: A mask or list of masks
        :return: the output of the classifier layer
        """
        # in contrast to other architectures, the input layer is skipped
        # here, because the backbone architecture has its own input handling
        resnet_1_out, resnet_2_out = self.backbone(inputs)

        aspp_out = self.aspp(resnet_2_out)  # usually resnet layer stg4 l6
        aspp_out = self.aspp_upsample(aspp_out)

        low_level_conv = self.low_level(resnet_1_out)  # usually resnet stg2 l3

        x = self.concat([aspp_out, low_level_conv])

        for layer in self.decoder_layers:
            x = layer(x)

        # softmax classifier head layer
        classes = self.outputs(x)

        return classes

    def instantiate_layers(self):
        """Instantiate layers lying between the input and the classifier."""
        # skipping last block of ResNet - seems to correspond with the
        # original DeepLabv3+ paper
        self.backbone = ResNet(self.nr_classes, pooling=self.resnet_pooling,
                               include_top=False, depth=self.resnet_depth,
                               activation=self.activation,
                               use_bias=self.use_bias,
                               dropout_rate_hidden=self.dropout_rate_hidden,
                               return_layers=('id_block_2_3',
                                              self.resnet_2_out),
                               name='resnet')

        backbone_out_1_pooled = 4
        if 'block_4' in self.resnet_2_out:
            backbone_out_2_pooled = 16
        elif 'block_5' in self.resnet_2_out:
            backbone_out_2_pooled = 32
        else:
            raise ModelConfigError('So far only id_block_4_6 and id_block_5_3 '
                                   'are supported as the deepest outputs from '
                                   'ResNet for DeepLabv3+')

        # following the paper in using only dilation rates 1, 6, 12, and 18
        # pool_dims should correspond to the dims of the returned layers from
        # the backbone model
        self.aspp = ASPP(
            dilation_rates=(1, 6, 12, 18),
            pool_dims=(self.tensor_shape[0] // backbone_out_2_pooled,
                       self.tensor_shape[1] // backbone_out_2_pooled),
            activation=self.activation,
            dropout_rate=self.dropout_rate_hidden)

        self.aspp_upsample = UpSampling2D(
            size=[backbone_out_2_pooled // backbone_out_1_pooled,
                  backbone_out_2_pooled // backbone_out_1_pooled],
            interpolation='bilinear',
            name='aspp_upsample')

        self.low_level = ConvBlock(filters=(48, ),
                                   kernel_sizes=((1, 1), ),
                                   activations=(self.activation,),
                                   dropout_rate=self.dropout_rate_hidden,
                                   paddings=('same',),
                                   depth=1,
                                   kernel_initializer='he_normal',
                                   name='low_level_conv_block',
                                   use_bias=self.use_bias)

        self.concat = Concatenate(name='decoder_concat')

        # decoder
        self.decoder_layers = [
            ConvBlock(filters=(256, 256),
                      kernel_sizes=((3, 3), ),
                      activations=(self.activation,),
                      paddings=('same',),
                      dropout_rate=self.dropout_rate_hidden,
                      depth=2,
                      kernel_initializer='he_normal',
                      name='decoder_conv_blocks',
                      use_bias=self.use_bias),
            UpSampling2D(size=[backbone_out_1_pooled,
                               backbone_out_1_pooled],
                         interpolation='bilinear',
                         name='decoder_final_upsample')]

    def get_config(self):
        """Return the configuration of the convolutional block.

        Allows later reinstantiation (without its trained weights) from this
        configuration. It does not include connectivity information, nor the
        model class name.

        :return: the configuration dictionary of the convolutional block
        """
        config = super(DeepLabv3Plus, self).get_config()

        config.update(resnet_pooling=self.resnet_pooling,
                      resnet_depth=self.resnet_depth,
                      resnet_2_out=self.resnet_2_out)

        return config


def create_model(model, nr_classes, nr_bands, tensor_shape,
                 nr_filters=64, optimizer='adam', loss='dice', metrics=None,
                 activation='relu', padding='same', verbose=1, alpha=None,
                 beta=None, dropout_rate_input=None, dropout_rate_hidden=None,
                 name='model'):
    """Create intended model.

    :param model: model architecture
    :param nr_classes: number of classes to be predicted
    :param nr_bands: number of bands of intended input images
    :param tensor_shape: shape of the first two dimensions of input tensors
    :param nr_filters: base number of convolution filters (multiplied deeper
        in the model)
    :param optimizer: name of built-in optimizer or optimizer instance
    :param loss: name of a built-in objective function,
        objective function or tf.keras.losses.Loss instance
    :param metrics: list of metrics to be evaluated by the model during
        training and testing. Each of this can be a name of a built-in
        function, function or a tf.keras.metrics.Metric instance
    :param activation: string name of built-in activation function, such as
        'relu'
    :param padding: 'valid' means no padding. 'same' results in padding
        evenly to the left/right or up/down of the input such that output
        has the same height/width dimension as the input
    :param verbose: verbosity (0=quiet, >0 verbose)
    :param alpha: magnitude of penalties for false positives for Tversky loss
    :param beta: magnitude of penalties for false negatives for Tversky loss
    :param dropout_rate_input: float between 0 and 1. Fraction of the input
        units of the input layer to drop
    :param dropout_rate_hidden: float between 0 and 1. Fraction of the input
        units of the hidden layers to drop
    :param name: The name of the model
    :return: compiled model
    """
    model_classes = {'U-Net': UNet, 'SegNet': SegNet, 'DeepLab': DeepLabv3Plus}
    activations = {'relu': k_layers.ReLU, 'leaky_relu': k_layers.LeakyReLU,
                   'prelu': k_layers.PReLU}

    if metrics is None:
        metrics = ['accuracy']

    model = model_classes[model](nr_classes, nr_bands=nr_bands,
                                 nr_filters=nr_filters,
                                 tensor_shape=tensor_shape,
                                 activation=activations[activation],
                                 padding=padding,
                                 dropout_rate_input=dropout_rate_input,
                                 dropout_rate_hidden=dropout_rate_hidden,
                                 name=name)

    # get loss functions corresponding to non-TF losses
    if loss == 'dice':
        loss = categorical_dice
    elif loss == 'tversky':
        loss = lambda gt, p: categorical_tversky(gt, p, alpha, beta)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.build(input_shape=(None, tensor_shape[0], tensor_shape[1], nr_bands))

    if verbose > 0:
        model.summary()

    return model
