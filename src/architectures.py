#!/usr/bin/python3

import tensorflow as tf

from abc import ABC, abstractmethod
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Input, UpSampling2D, \
    Concatenate, Dropout, ZeroPadding2D, BatchNormalization, Activation, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, add, AveragePooling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from cnn_lib import ConvBlock, MyMaxPooling, MyMaxUnpooling, \
    categorical_dice, categorical_tversky, ResBlock
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
            us_concats.append(Concatenate(axis=3, name=f'upsampling_concat{i}'))
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


def create_model(model, nr_classes, nr_bands, tensor_shape,
                 nr_filters=64, optimizer='adam', loss='dice', metrics=None,
                 activation='relu', padding='same', verbose=1, alpha=None,
                 beta=None, dropout_rate_input=None, dropout_rate_hidden=None):
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
    :param activation: activation function, such as tf.nn.relu, or string
        name of built-in activation function, such as 'relu'
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
    :return: compiled model
    """
    model_classes = {'U-Net': UNet, 'SegNet': SegNet, 'DeepLab': DeepLabV3Plus}

    if metrics is None:
        metrics = ['accuracy']

    model = model_classes[model](nr_classes, nr_bands=nr_bands,
                                 nr_filters=nr_filters,
                                 tensor_shape=tensor_shape,
                                 activation=activation, padding=padding,
                                 dropout_rate_input=dropout_rate_input,
                                 dropout_rate_hidden=dropout_rate_hidden)

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

# _OFFSET_OUTPUT = 'offset'
#
# def create_encoder(backbone_options='resnet',
#                    bn_layer: tf.keras.layers.Layer,
#                    conv_kernel_weight_decay: float = 0.0) -> tf.keras.Model:
#   """Creates an encoder.
#   Args:
#     backbone_options: A proto config of type
#       config_pb2.ModelOptions.BackboneOptions.
#     bn_layer: A tf.keras.layers.Layer that computes the normalization.
#     conv_kernel_weight_decay: A float, the weight decay for convolution kernels.
#   Returns:
#     An instance of tf.keras.Model containing the encoder.
#   Raises:
#     ValueError: An error occurs when the specified encoder meta architecture is
#       not supported.
#   """
#   if ('resnet' in backbone_options.name or
#       'swidernet' in backbone_options.name or
#       'axial_deeplab' in backbone_options.name or
#       'max_deeplab' in backbone_options.name):
#       return create_resnet_encoder(
#           backbone_options,
#           bn_layer=bn_layer,
#           conv_kernel_weight_decay=conv_kernel_weight_decay)
#   # elif 'mobilenet' in backbone_options.name:
#   #   return create_mobilenet_encoder(
#   #       backbone_options,
#   #       bn_layer=bn_layer,
#   #       conv_kernel_weight_decay=conv_kernel_weight_decay)
#   # raise ValueError('The specified encoder %s is not a valid encoder.' %
#   #                  backbone_options.name)
#
# class AxialResNetInstance(axial_resnet.AxialResNet):  # pytype: disable=ignored-abstractmethod  # abcmeta-check
#   """A base Axial-ResNet model."""
#
#   @classmethod
#   @abc.abstractmethod
#   def _get_config(cls):
#     pass
#
#   def __init__(self, name, **kwargs):
#     """Builds an Axial-ResNet model."""
#     # Get the config of the current model.
#     current_config = self._get_config()
#
#     # Override the default config with the current config. This line can be
#     # omitted because the default config equals the default arguments of the
#     # functions that build the model. But we make all the configs explicit here.
#     current_config = override(_get_default_config(), current_config)
#
#     # Finally, override the current model config with keyword arguments. In this
#     # way, we still respect arguments passed as keyword arguments, such as
#     # classification_mode, output_stride, etc.
#     current_config = override(current_config, kwargs)
#     logging.info('Axial-ResNet final config: %s', current_config)
#     super(AxialResNetInstance, self).__init__(name, **current_config)
#
# def create_resnet_encoder(
#     backbone_options: config_pb2.ModelOptions.BackboneOptions,
#     bn_layer: tf.keras.layers.Layer,
#     conv_kernel_weight_decay: float = 0.0) -> tf.keras.Model:
#   """Creates a ResNet encoder specified by name.
#   Args:
#     backbone_options: A proto config of type
#       config_pb2.ModelOptions.BackboneOptions.
#     bn_layer: A tf.keras.layers.Layer that computes the normalization.
#     conv_kernel_weight_decay: A float, the weight decay for convolution kernels.
#   Returns:
#     An instance of tf.keras.Model containing the ResNet encoder.
#   """
#   return axial_resnet_instances.get_model(
#       backbone_options.name,
#       output_stride=backbone_options.output_stride,
#       stem_width_multiplier=backbone_options.stem_width_multiplier,
#       width_multiplier=backbone_options.backbone_width_multiplier,
#       backbone_layer_multiplier=backbone_options.backbone_layer_multiplier,
#       block_group_config={
#           'use_squeeze_and_excite': backbone_options.use_squeeze_and_excite,
#           'drop_path_keep_prob': backbone_options.drop_path_keep_prob,
#           'drop_path_schedule': backbone_options.drop_path_schedule,
#           'use_sac_beyond_stride': backbone_options.use_sac_beyond_stride},
#       bn_layer=bn_layer,
#       conv_kernel_weight_decay=conv_kernel_weight_decay)
#
# class DeepLab(tf.keras.Model):
#   """This class represents the DeepLab meta architecture.
#   This class supports four architectures of the DeepLab family: DeepLab V3,
#   DeepLab V3+, Panoptic-DeepLab, and MaX-DeepLab. The exact architecture must be
#   defined during initialization.
#   """
#
#   def __init__(self,
#                config: config_pb2.ExperimentOptions,
#                dataset_descriptor: dataset.DatasetDescriptor):
#     """Initializes a DeepLab architecture.
#     Args:
#       config: A config_pb2.ExperimentOptions configuration.
#       dataset_descriptor: A dataset.DatasetDescriptor.
#     Raises:
#       ValueError: If MaX-DeepLab is used with multi-scale inference.
#     """
#     super(DeepLab, self).__init__(name='DeepLab')
#
#     # if config.trainer_options.solver_options.use_sync_batchnorm:
#     #   logging.info('Synchronized Batchnorm is used.')
#     #   bn_layer = functools.partial(
#     #       tf.keras.layers.experimental.SyncBatchNormalization,
#     #       momentum=config.trainer_options.solver_options.batchnorm_momentum,
#     #       epsilon=config.trainer_options.solver_options.batchnorm_epsilon)
#     # else:
#     #   logging.info('Standard (unsynchronized) Batchnorm is used.')
#     bn_layer = functools.partial(
#         tf.keras.layers.BatchNormalization,
#         momentum=config.trainer_options.solver_options.batchnorm_momentum,
#         epsilon=config.trainer_options.solver_options.batchnorm_epsilon)
#
#     # Divide weight decay by 2 to match the implementation of tf.nn.l2_loss. In
#     # this way, we allow our users to use a normal weight decay (e.g., 1e-4 for
#     # ResNet variants) in the config textproto. Then, we pass the adjusted
#     # weight decay (e.g., 5e-5 for ResNets) to keras in order to exactly match
#     # the commonly used tf.nn.l2_loss in TF1. References:
#     # https://github.com/tensorflow/models/blob/68ee72ae785274156b9e943df4145b257cd78b32/official/vision/beta/tasks/image_classification.py#L41
#     # https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2
#     # https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
#     self._encoder = builder.create_encoder(
#         config.model_options.backbone, bn_layer,
#         conv_kernel_weight_decay=(
#             config.trainer_options.solver_options.weight_decay / 2))
#
#     self._decoder = builder.create_decoder(
#         config.model_options, bn_layer, dataset_descriptor.ignore_label)
#
#     self._is_max_deeplab = (
#         config.model_options.WhichOneof('meta_architecture') == 'max_deeplab')
#     self._post_processor = post_processor_builder.get_post_processor(
#         config, dataset_descriptor)
#
#     # The ASPP pooling size is always set to train crop size, which is found to
#     # be experimentally better.
#     pool_size = config.train_dataset_options.crop_size
#     output_stride = float(config.model_options.backbone.output_stride)
#     pool_size = tuple(
#         utils.scale_mutable_sequence(pool_size, 1.0 / output_stride))
#     logging.info('Setting pooling size to %s', pool_size)
#     self.set_pool_size(pool_size)
#
#     # Variables for multi-scale inference.
#     self._add_flipped_images = config.evaluator_options.add_flipped_images
#     if not config.evaluator_options.eval_scales:
#       self._eval_scales = [1.0]
#     else:
#       self._eval_scales = config.evaluator_options.eval_scales
#     if self._is_max_deeplab and (
#         self._add_flipped_images or len(self._eval_scales) > 1):
#       raise ValueError(
#           'MaX-DeepLab does not support multi-scale inference yet.')
#
#   def call(self,
#            input_tensor: tf.Tensor,
#            training: bool = False) -> Dict[Text, Any]:
#     """Performs a forward pass.
#     Args:
#       input_tensor: An input tensor of type tf.Tensor with shape [batch, height,
#         width, channels]. The input tensor should contain batches of RGB images.
#       training: A boolean flag indicating whether training behavior should be
#         used (default: False).
#     Returns:
#       A dictionary containing the results of the specified DeepLab architecture.
#       The results are bilinearly upsampled to input size before returning.
#     """
#     # Normalize the input in the same way as Inception. We normalize it outside
#     # the encoder so that we can extend encoders to different backbones without
#     # copying the normalization to each encoder. We normalize it after data
#     # preprocessing because it is faster on TPUs than on host CPUs. The
#     # normalization should not increase TPU memory consumption because it does
#     # not require gradient.
#     input_tensor = input_tensor / 127.5 - 1.0
#     # Get the static spatial shape of the input tensor.
#     _, input_h, input_w, _ = input_tensor.get_shape().as_list()
#     if training:
#       result_dict = self._decoder(
#           self._encoder(input_tensor, training=training), training=training)
#       result_dict = self._resize_predictions(
#           result_dict,
#           target_h=input_h,
#           target_w=input_w)
#     else:
#       result_dict = collections.defaultdict(list)
#       # Evaluation mode where one could perform multi-scale inference.
#       scale_1_pool_size = self.get_pool_size()
#       logging.info('Eval with scales %s', self._eval_scales)
#       for eval_scale in self._eval_scales:
#         # Get the scaled images/pool_size for each scale.
#         scaled_images, scaled_pool_size = (
#             self._scale_images_and_pool_size(
#                 input_tensor, list(scale_1_pool_size), eval_scale))
#         # Update the ASPP pool size for different eval scales.
#         self.set_pool_size(tuple(scaled_pool_size))
#         logging.info('Eval scale %s; setting pooling size to %s',
#                      eval_scale, scaled_pool_size)
#         pred_dict = self._decoder(
#             self._encoder(scaled_images, training=training), training=training)
#         # MaX-DeepLab skips this resizing and upsamples the mask outputs in
#         # self._post_processor.
#         pred_dict = self._resize_predictions(
#             pred_dict,
#             target_h=input_h,
#             target_w=input_w)
#         # Change the semantic logits to probabilities with softmax. Note
#         # one should remove semantic logits for faster inference. We still
#         # keep them since they will be used to compute evaluation loss.
#         pred_dict[common.PRED_SEMANTIC_PROBS_KEY] = tf.nn.softmax(
#             pred_dict[common.PRED_SEMANTIC_LOGITS_KEY])
#         # Store the predictions from each scale.
#         for output_type, output_value in pred_dict.items():
#           result_dict[output_type].append(output_value)
#         if self._add_flipped_images:
#           pred_dict_reverse = self._decoder(
#               self._encoder(tf.reverse(scaled_images, [2]), training=training),
#               training=training)
#           pred_dict_reverse = self._resize_predictions(
#               pred_dict_reverse,
#               target_h=input_h,
#               target_w=input_w,
#               reverse=True)
#           # Change the semantic logits to probabilities with softmax.
#           pred_dict_reverse[common.PRED_SEMANTIC_PROBS_KEY] = tf.nn.softmax(
#               pred_dict_reverse[common.PRED_SEMANTIC_LOGITS_KEY])
#           # Store the predictions from each scale.
#           for output_type, output_value in pred_dict_reverse.items():
#             result_dict[output_type].append(output_value)
#       # Set back the pool_size for scale 1.0, the original setting.
#       self.set_pool_size(tuple(scale_1_pool_size))
#       # Average results across scales.
#       for output_type, output_value in result_dict.items():
#         result_dict[output_type] = tf.reduce_mean(
#             tf.stack(output_value, axis=0), axis=0)
#       # Post-process the results.
#       result_dict.update(self._post_processor(result_dict))
#
#     if common.PRED_CENTER_HEATMAP_KEY in result_dict:
#       result_dict[common.PRED_CENTER_HEATMAP_KEY] = tf.squeeze(
#           result_dict[common.PRED_CENTER_HEATMAP_KEY], axis=3)
#     return result_dict
#
#   def reset_pooling_layer(self):
#     """Resets the ASPP pooling layer to global average pooling."""
#     self._decoder.reset_pooling_layer()
#
#   def set_pool_size(self, pool_size: Tuple[int, int]):
#     """Sets the pooling size of the ASPP pooling layer.
#     Args:
#       pool_size: A tuple specifying the pooling size of the ASPP pooling layer.
#     """
#     self._decoder.set_pool_size(pool_size)
#
#   def get_pool_size(self):
#     return self._decoder.get_pool_size()
#
#   @property
#   def checkpoint_items(self) -> Dict[Text, Any]:
#     items = dict(encoder=self._encoder)
#     items.update(self._decoder.checkpoint_items)
#     return items
#
#   def _resize_predictions(self, result_dict, target_h, target_w, reverse=False):
#     """Resizes predictions to the target height and width.
#     This function resizes the items in the result_dict to the target height and
#     width. The items are optionally reversed w.r.t width if `reverse` is True.
#     Args:
#       result_dict: A dictionary storing prediction results to be resized.
#       target_h: An integer, the target height.
#       target_w: An integer, the target width.
#       reverse: A boolean, reversing the prediction result w.r.t. width.
#     Returns:
#       Resized (or optionally reversed) result_dict.
#     """
#     # The default MaX-DeepLab paper does not upsample any output during training
#     # in order to save GPU/TPU memory, but upsampling might lead to better
#     # performance.
#     if self._is_max_deeplab:
#       return result_dict
#     for key, value in result_dict.items():
#       if reverse:
#         value = tf.reverse(value, [2])
#         # Special care to offsets: need to flip x-offsets.
#         if _OFFSET_OUTPUT in key:
#           offset_y, offset_x = tf.split(
#               value=value, num_or_size_splits=2, axis=3)
#           offset_x *= -1
#           value = tf.concat([offset_y, offset_x], 3)
#       if _OFFSET_OUTPUT in key:
#         result_dict[key] = utils.resize_and_rescale_offsets(
#             value, [target_h, target_w])
#       else:
#         result_dict[key] = utils.resize_bilinear(
#             value, [target_h, target_w])
#     return result_dict
#
#   def _scale_images_and_pool_size(self, images, pool_size, scale):
#     """Scales images and pool_size w.r.t. scale.
#     Args:
#       images: An input tensor with shape [batch, height, width, 3].
#       pool_size: A list with two elements, specifying the pooling size
#         of ASPP pooling layer.
#       scale: A float, used to scale the input images and pool_size.
#     Returns:
#       Scaled images, and pool_size.
#     """
#     if scale == 1.0:
#       scaled_images = images
#       scaled_pool_size = pool_size
#     else:
#       image_size = images.get_shape().as_list()[1:3]
#       scaled_image_size = utils.scale_mutable_sequence(image_size, scale)
#       scaled_images = utils.resize_bilinear(images, scaled_image_size)
#       scaled_pool_size = [None, None]
#       if pool_size != [None, None]:
#         scaled_pool_size = utils.scale_mutable_sequence(pool_size, scale)
#     return scaled_images, scaled_pool_size












def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=True,
             # weights='imagenet',
             weights=None,
             input_tensor=None,
             input_shape=None,
             # pooling=None,
             pooling='avg',
             classes=1000,
             **kwargs):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    For the original paper, see <https://arxiv.org/pdf/1512.03385.pdf>
    ONDRA: Change from the original paper - batch norm used after activation
    functions, not before
    # motivation: <https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/>
    # motivation: <https://blog.paperspace.com/busting-the-myths-about-batch-normalization/>
    # motivation: <https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout>

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    inputs = Input(shape=input_shape)
    bn_axis = 3

    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(inputs)
    x = Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = ResBlock(kernel_size=3, filters=(64, 64, 256), strides=(1, 1),
                 name='res_block_2_a')(x)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = ResBlock(kernel_size=3, filters=(128, 128, 512),
                 name='res_block_3_a')(x)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = ResBlock(kernel_size=3, filters=(256, 256, 1024),
                 name='res_block_4_a')(x)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = ResBlock(kernel_size=3, filters=(512, 512, 2048),
                 name='res_block_5_a')(x)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Create model.
    model = Model(inputs, x, name='resnet50')

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

# import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import AveragePooling2D, Lambda, Conv2D, \
#     Conv2DTranspose, Activation, Reshape, concatenate, Concatenate, \
#     BatchNormalization, ZeroPadding2D

def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError('Invalid keyword argument: %s', key)
    return backend, layers, models, utils


def decode_predictions(preds, top=5, **kwargs):
    """Decodes the prediction of an ImageNet model.
    # Arguments
        preds: Numpy tensor encoding a batch of predictions.
        top: Integer, how many top-guesses to return.
    # Returns
        A list of lists of top class prediction tuples
        `(class_name, class_description, score)`.
        One list of tuples per sample in batch input.
    # Raises
        ValueError: In case of invalid shape of the `pred` array
            (must be 2D).
    """
    global CLASS_INDEX

    backend, _, _, keras_utils = get_submodules_from_kwargs(kwargs)

    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = keras_utils.get_file(
            'imagenet_class_index.json',
            CLASS_INDEX_PATH,
            cache_subdir='models',
            file_hash='c2c37ea517e94d9795004a39431a14cb')
        with open(fpath) as f:
            CLASS_INDEX = json.load(f)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        require_flatten,
                        weights=None):
    """Internal utility to compute/validate a model's input shape.
    # Arguments
        input_shape: Either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_size: Default input width/height for the model.
        min_size: Minimum input width/height accepted by the model.
        data_format: Image data format to use.
        require_flatten: Whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: One of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
            If weights='imagenet' input channels must be equal to 3.
    # Returns
        An integer shape tuple (may include None entries).
    # Raises
        ValueError: In case of invalid argument values.
    """
    if len(input_shape) != 3:
        raise ModelConfigError(
            '`input_shape` must be a tuple of three integers.')

    if data_format == 'channels_first':
        if ((input_shape[1] is not None and input_shape[1] < min_size) or
           (input_shape[2] is not None and input_shape[2] < min_size)):
            raise ValueError('Input size must be at least ' +
                             str(min_size) + 'x' + str(min_size) +
                             '; got `input_shape=' +
                             str(input_shape) + '`')
    else:
        if ((input_shape[0] is not None and input_shape[0] < min_size) or
           (input_shape[1] is not None and input_shape[1] < min_size)):
            raise ValueError('Input size must be at least ' +
                             str(min_size) + 'x' + str(min_size) +
                             '; got `input_shape=' +
                             str(input_shape) + '`')

    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape


def ASPP(tensor):
    '''atrous spatial pyramid pooling'''
    dims = K.int_shape(tensor)

    y_pool = AveragePooling2D(pool_size=(
        dims[1], dims[2]), name='average_pooling')(tensor)
    y_pool = Conv2D(filters=256, kernel_size=1, padding='same',
                    kernel_initializer='he_normal', name='pool_1x1conv2d',
                    use_bias=False)(y_pool)
    y_pool = BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = Activation('relu', name=f'relu_1')(y_pool)

    y_pool = UpSampling2D(size=[dims[1] // y_pool.shape[1],
                                dims[2] // y_pool.shape[2]],
                          interpolation='bilinear')(y_pool)

    y_1 = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_d1',
                 use_bias=False)(tensor)
    y_1 = BatchNormalization(name=f'bn_2')(y_1)
    y_1 = Activation('relu', name=f'relu_2')(y_1)

    y_6 = Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_d6',
                 use_bias=False)(tensor)
    y_6 = BatchNormalization(name=f'bn_3')(y_6)
    y_6 = Activation('relu', name=f'relu_3')(y_6)

    y_12 = Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_d12',
                  use_bias=False)(tensor)
    y_12 = BatchNormalization(name=f'bn_4')(y_12)
    y_12 = Activation('relu', name=f'relu_4')(y_12)

    y_18 = Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_d18',
                  use_bias=False)(tensor)
    y_18 = BatchNormalization(name=f'bn_5')(y_18)
    y_18 = Activation('relu', name=f'relu_5')(y_18)

    y = concatenate([y_pool, y_1, y_6, y_12, y_18], name='ASPP_concat')

    y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
               kernel_initializer='he_normal', name='ASPP_conv2d_final',
               use_bias=False)(y)
    y = BatchNormalization(name=f'bn_final')(y)
    y = Activation('relu', name=f'relu_final')(y)
    return y

import tensorflow.keras.layers, tensorflow.keras.models, tensorflow.keras.utils
_KERAS_BACKEND = K
_KERAS_LAYERS = tensorflow.keras.layers
_KERAS_MODELS = tensorflow.keras.models
_KERAS_UTILS = tensorflow.keras.utils


# def DeepLabV3Plus(img_height, img_width, nclasses=66):
def DeepLabV3Plus(nclasses, img_height=512, img_width=512, **kwargs):
    """<https://arxiv.org/pdf/1802.02611.pdf>"""
    print('*** Building DeepLabv3Plus Network ***')

    base_model = ResNet50(input_shape=(
        # img_height, img_width, 3), weights='imagenet', include_top=False)
        img_height, img_width, kwargs['nr_bands']), weights=None,
        include_top=False)

    image_features = base_model.get_layer('activation_39').output
    x_a = ASPP(image_features)
    x_a = UpSampling2D(size=[img_height // 4 // x_a.shape[1],
                             img_width // 4 // x_a.shape[2]],
                       interpolation='bilinear')(x_a)

    x_b = base_model.get_layer('activation_9').output
    x_b = Conv2D(filters=48, kernel_size=1, padding='same',
                 kernel_initializer='he_normal', name='low_level_projection',
                 use_bias=False)(x_b)
    x_b = BatchNormalization(name=f'bn_low_level_projection')(x_b)
    x_b = Activation('relu', name='low_level_activation')(x_b)

    x = concatenate([x_a, x_b], name='decoder_concat')

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
               kernel_initializer='he_normal', name='decoder_conv2d_1',
               use_bias=False)(x)
    x = BatchNormalization(name=f'bn_decoder_1')(x)
    x = Activation('relu', name='activation_decoder_1')(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
               kernel_initializer='he_normal', name='decoder_conv2d_2',
               use_bias=False)(x)
    x = BatchNormalization(name=f'bn_decoder_2')(x)
    x = Activation('relu', name='activation_decoder_2')(x)
    x = UpSampling2D(size=[img_height // x.shape[1], img_width // x.shape[2]],
                     interpolation='bilinear')(x)

    x = Conv2D(nclasses, (1, 1), name='output_layer')(x)
    '''
    x = Activation('softmax')(x) 
    tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    Args:
        from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
        we assume that `y_pred` encodes a probability distribution.
    '''
    model = Model(inputs=base_model.input, outputs=x, name='DeepLabV3_Plus')
    print(f'*** Output_Shape => {model.output_shape} ***')
    return model
