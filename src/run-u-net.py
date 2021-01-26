#!/usr/bin/python3

import os
import glob
import argparse

import numpy as np
import tensorflow as tf

from osgeo import gdal
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, \
    EarlyStopping

from data_preparation import parse_label_code
from cnn_lib import AugmentGenerator, categorical_dice, categorical_tversky
from architectures import get_unet
from visualization import write_stats, visualize_detections


def main(operation, data_dir, output_dir, model_fn, in_model_path,
         visualization_path, nr_epochs, initial_epoch, batch_size,
         loss_function, seed, patience, tensor_shape, monitored_value,
         force_dataset_generation, tversky_alpha, tversky_beta,
         dropout_rate_input, dropout_rate_hidden):
    print_device_info()

    # get nr of bands
    dataset = glob.glob(os.path.join(data_dir, '*[0-9].tif'))
    dataset_image = gdal.Open(dataset[0])
    nr_bands = dataset_image.RasterCount
    dataset_image = None

    label_codes, label_names, id2code = get_codings(
        os.path.join(data_dir, 'label_colors.txt'))

    # set TensorFlow seed
    tf.random.set_seed(seed)

    model = create_model(
        len(id2code), nr_bands, tensor_shape, loss=loss_function,
        alpha=tversky_alpha, beta=tversky_beta,
        dropout_rate_input=dropout_rate_input,
        dropout_rate_hidden=dropout_rate_hidden)

    # val generator used for both the training and the detection
    val_generator = AugmentGenerator(data_dir, batch_size, 'val', nr_bands,
                                     tensor_shape, force_dataset_generation)

    # load weights if the model is supposed to do so
    if operation in ('detect', 'fine-tune'):
        model.load_weights(in_model_path)

    if operation in ('train', 'fine-tune'):
        # Train or fine-tune model
        train_generator = AugmentGenerator(data_dir, batch_size, 'train')
        train(model, train_generator, val_generator, id2code, batch_size,
              output_dir, visualization_path, model_fn, nr_epochs,
              initial_epoch, seed=seed, patience=patience,
              monitored_value=monitored_value)
    else:
        # detect
        detect(model, val_generator, id2code, batch_size,
               [i[0] for i in label_codes], label_names, seed,
               visualization_path)


def print_device_info():
    """Print info about used GPUs."""
    print('Available GPUs:')
    print(tf.config.list_physical_devices('GPU'))

    print('Device name:')
    print(tf.random.uniform((1, 1)).device)

    print('TF executing eagerly:')
    print(tf.executing_eagerly())


def get_codings(description_file):
    """Get lists of label codes and names and a an id-name mapping dictionary.

    :param description_file: path to the txt file with labels and their names
    :return: list of label codes, list of label names, id2code dictionary
    """
    label_codes, label_names = zip(
        *[parse_label_code(i) for i in open(description_file)])
    label_codes, label_names = list(label_codes), list(label_names)
    id2code = {i: j for i, j in enumerate(label_codes)}

    return label_codes, label_names, id2code


def create_model(nr_classes, nr_bands, tensor_shape, optimizer='adam',
                 loss='dice', metrics=None, activation='relu',
                 padding='same', verbose=1, alpha=None, beta=None,
                 dropout_rate_input=None, dropout_rate_hidden=None):
    """Create intended model.

    So far it is only U-Net.

    :param nr_classes: number of classes to be predicted
    :param nr_bands: number of bands of intended input images
    :param tensor_shape: shape of the first two dimensions of input tensors
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
    if metrics is None:
        metrics = ['accuracy']

    model = get_unet(nr_classes, nr_bands=nr_bands, nr_filters=32,
                     tensor_shape=tensor_shape, activation=activation,
                     padding=padding, dropout_rate_input=dropout_rate_input,
                     dropout_rate_hidden=dropout_rate_hidden)

    # get loss functions corresponding to non-TF losses
    if loss == 'dice':
        loss = categorical_dice
    elif loss == 'tversky':
        loss = lambda gt, p: categorical_tversky(gt, p, alpha, beta)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    if verbose > 0:
        model.summary()

    return model


def train(model, train_generator, val_generator, id2code, batch_size,
          output_dir, visualization_path, model_fn, nr_epochs,
          initial_epoch=0, seed=1, patience=100,
          monitored_value='val_accuracy'):
    """Run model training.

    :param model: model to be used for the detection
    :param train_generator: training data generator
    :param val_generator: validation data generator
    :param id2code: dictionary mapping label ids to their codes
    :param batch_size: the number of samples that will be propagated through
        the network at once
    :param output_dir: path where logs and the model will be saved
    :param visualization_path: path to a directory where the output
        visualizations will be saved
    :param model_fn: model file name
    :param nr_epochs: number of epochs to train the model
    :param initial_epoch: epoch at which to start training
    :param seed: the generator seed
    :param patience: number of epochs with no improvement after which training
        will be stopped
    :param monitored_value: metric name to be monitored
    """
    # set up model_path
    if model_fn is None:
        model_fn = 'lc_ep{}_bs{}.h5'.format(args.nr_epochs, args.batch_size)
    else:
        model_fn = args.model_fn

    out_model_path = os.path.join(output_dir, model_fn)

    # set up log dir
    log_dir = os.path.join(output_dir, 'logs')

    # create output_dir if it does not exist
    if not os.path.exists(output_dir):
        os.mkdir(args.output_dir)

    # set up monitoring
    tb = TensorBoard(log_dir=log_dir, write_graph=True)
    mc = ModelCheckpoint(
        mode='max', filepath=out_model_path,
        monitor=monitored_value, save_best_only='True',
        save_weights_only='True',
        verbose=1)
    # TODO: check custom earlystopping to monitor multiple metrics
    #       https://stackoverflow.com/questions/64556120/early-stopping-with-multiple-conditions
    es = EarlyStopping(mode='max', monitor='val_accuracy', patience=patience,
                       verbose=1, restore_best_weights=True)
    callbacks = [tb, mc, es]

    # TODO: parameterize?
    steps_per_epoch = np.ceil(train_generator.nr_samples / batch_size)
    validation_steps = np.ceil(val_generator.nr_samples / batch_size)

    # train
    result = model.fit(
        train_generator(id2code, seed),
        validation_data=val_generator(id2code, seed),
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=nr_epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks)

    write_stats(result, os.path.join(visualization_path, 'accu.png'))


def detect(model, val_generator, id2code, batch_size,
           label_codes, label_names, seed=1, out_dir='/tmp'):
    """Run detection.

    :param model: model to be used for the detection
    :param val_generator: validation data generator
    :param id2code: dictionary mapping label ids to their codes
    :param batch_size: the number of samples that will be propagated through
        the network at once
    :param label_codes: list with label codes
    :param label_names: list with label names
    :param seed: the generator seed
    :param out_dir: directory where the output visualizations will be saved
    """
    # TODO: Do not test on augmented data
    testing_gen = val_generator(id2code, seed)

    batch_img, batch_mask = next(testing_gen)
    pred_all = model.predict(batch_img)

    visualize_detections(batch_img, batch_mask, pred_all, id2code,
                         label_codes, label_names, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run U-Net training and detection')

    parser.add_argument(
        '--operation', type=str, required=True,
        choices=('train', 'detect', 'fine-tune'),
        help='Choose either to train the model or to use a trained one for '
             'detection')
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Path to the directory containing images and labels')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Path where logs and the model will be saved')
    parser.add_argument(
        '--model_fn', type=str,
        help='ONLY FOR OPERATION == TRAIN: Output model filename')
    parser.add_argument(
        '--model_path', type=str, default=None,
        help='ONLY FOR OPERATION IN (DETECT, FINE-TUNE): Input model path')
    parser.add_argument(
        '--visualization_path', type=str, default='/tmp',
        help='Path to a directory where the accuracy visualization '
             '(operation == train) or detection visualizations '
             '(operation == detect) will be saved')
    parser.add_argument(
        '--nr_epochs', type=int, default=1,
        help='ONLY FOR OPERATION == TRAIN: Number of epochs to train '
             'the model. Note that in conjunction with initial_epoch, '
             'epochs is to be understood as the final epoch')
    parser.add_argument(
        '--initial_epoch', type=int, default=0,
        help='ONLY FOR OPERATION == FINE-TUNE: Epoch at which to start '
             'training (useful for resuming a previous training run)')
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='The number of samples that will be propagated through the '
             'network at once')
    parser.add_argument(
        '--loss_function', type=str, default='dice',
        choices=('dice', 'categorical_crossentropy', 'tversky'),
        help='A function that maps the training onto a real number '
             'representing cost associated with the epoch')
    parser.add_argument(
        '--seed', type=int, default=1,
        help='Generator random seed')
    parser.add_argument(
        '--patience', type=int, default=100,
        help='ONLY FOR OPERATION == TRAIN: Number of epochs with no '
             'improvement after which training will be stopped')
    parser.add_argument(
        '--tensor_height', type=int, default=256,
        help='Height of the tensor representing the image')
    parser.add_argument(
        '--tensor_width', type=int, default=256,
        help='Width of the tensor representing the image')
    parser.add_argument(
        '--monitored_value', type=str, default='val_accuracy',
        help='ONLY FOR OPERATION == TRAIN: Metric name to be monitored')
    parser.add_argument(
        '--force_dataset_generation', type=bool, default=False,
        help='Boolean to force the dataset structure generation')
    parser.add_argument(
        '--tversky_alpha', type=float, default=None,
        help='ONLY FOR LOSS_FUNCTION == TVERSKY: Coefficient alpha')
    parser.add_argument(
        '--tversky_beta', type=float, default=None,
        help='ONLY FOR LOSS_FUNCTION == TVERSKY: Coefficient beta')
    parser.add_argument(
        '--dropout_rate_input', type=float, default=None,
        help='ONLY FOR OPERATION == TRAIN: Fraction of the input units of the '
             'input layer to drop')
    parser.add_argument(
        '--dropout_rate_hidden', type=float, default=None,
        help='ONLY FOR OPERATION == TRAIN: Fraction of the input units of the '
             'hidden layers to drop')

    args = parser.parse_args()

    # check required arguments by individual operations
    if args.operation == 'train' and args.output_dir is None:
        raise parser.error(
            'Argument output_dir required for operation == train')
    if args.operation in ('detect', 'fine-tune') and args.model_path is None:
        raise parser.error(
            'Argument model_path required for operation in '
            '(detect, fine-tune)')
    if args.operation == 'train' and args.initial_epoch != 0:
        raise parser.error(
            'Argument initial_epoch must be 0 for operation == train')
    tversky_none = None in (args.tversky_alpha, args.tversky_beta)
    if args.loss_function == 'tversky' and tversky_none is True:
        raise parser.error(
            'Arguments tversky_alpha and tversky_beta must be set for '
            'loss_function == tversky')
    dropout_specified = args.dropout_rate_input is not None or \
                        args.dropout_rate_hidden is not None
    if args.operation != 'train' and dropout_specified is True:
        raise parser.error(
            'Dropout can be specified only for operation == train')

    main(args.operation, args.data_dir, args.output_dir, args.model_fn,
         args.model_path, args.visualization_path, args.nr_epochs,
         args.initial_epoch, args.batch_size, args.loss_function, args.seed,
         args.patience, (args.tensor_height, args.tensor_width),
         args.monitored_value, args.force_dataset_generation,
         args.tversky_alpha, args.tversky_beta, args.dropout_rate_input,
         args.dropout_rate_hidden)
