#!/usr/bin/python3

import os
import argparse

import numpy as np
import tensorflow as tf

from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, \
    EarlyStopping

# imports from this package
import utils

from cnn_lib import AugmentGenerator
from architectures import create_model
from visualization import write_stats


def main(operation, data_dir, output_dir, model, model_fn, in_weights_path=None,
         visualization_path='/tmp', nr_epochs=1, initial_epoch=0, batch_size=1,
         loss_function='dice', seed=1, patience=100, tensor_shape=(256, 256), monitored_value='val_accuracy',
         force_dataset_generation=False, fit_memory=False, augment=False,
         tversky_alpha=None, tversky_beta=None, dropout_rate_input=None,
         dropout_rate_hidden=None, val_set_pct=0.2, filter_by_class=None):
    tf.config.threading.set_inter_op_parallelism_threads(3)
    N = 3
    os.environ["OMP_NUM_THREADS"] = f"{N}"
    os.environ['TF_NUM_INTEROP_THREADS'] = f"{N}"
    os.environ['TF_NUM_INTRAOP_THREADS'] = f"{N}"
    utils.print_device_info()

    # get nr of bands
    nr_bands = utils.get_nr_of_bands(data_dir)

    label_codes, label_names, id2code = utils.get_codings(
        os.path.join(data_dir, 'label_colors.txt'))

    # set TensorFlow seed
    tf.random.set_seed(seed)

    model = create_model(
        model, len(id2code), nr_bands, tensor_shape, loss=loss_function,
        alpha=tversky_alpha, beta=tversky_beta,
        dropout_rate_input=dropout_rate_input,
        dropout_rate_hidden=dropout_rate_hidden)

    # val generator used for both the training and the detection
    val_generator = AugmentGenerator(
        data_dir, batch_size, 'val', tensor_shape, force_dataset_generation,
        fit_memory, augment=augment, val_set_pct=val_set_pct,
        filter_by_class=filter_by_class)

    # load weights if the model is supposed to do so
    if operation == 'fine-tune':
        model.load_weights(in_weights_path)

    train_generator = AugmentGenerator(
        data_dir, batch_size, 'train', fit_memory=fit_memory,
        augment=augment)
    train(model, train_generator, val_generator, id2code, batch_size,
          output_dir, visualization_path, model_fn, nr_epochs,
          initial_epoch, seed=seed, patience=patience,
          monitored_value=monitored_value)


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
        model_fn = '{}_ep{}_pat{}.h5'.format(model.lower(), nr_epochs,
                                             patience)

    out_model_path = os.path.join(output_dir, model_fn)

    # set up log dir
    log_dir = os.path.join(output_dir, 'logs')

    # create output_dir if it does not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # get the correct early stop mode
    if 'accuracy' in monitored_value:
        early_stop_mode = 'max'
    else:
        early_stop_mode = 'min'

    # set up monitoring
    tb = TensorBoard(log_dir=log_dir, write_graph=True)
    mc = ModelCheckpoint(
        mode=early_stop_mode, filepath=out_model_path,
        monitor=monitored_value, save_best_only='True',
        save_weights_only='True',
        verbose=1)
    # TODO: check custom earlystopping to monitor multiple metrics
    #       https://stackoverflow.com/questions/64556120/early-stopping-with-multiple-conditions
    es = EarlyStopping(mode=early_stop_mode, monitor=monitored_value,
                       patience=patience, verbose=1, restore_best_weights=True)
    callbacks = [tb, mc, es]

    # steps per epoch not needed to be specified if the data are augmented, but
    # not when they are not (our own generator is used)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run training or fine-tuning')

    parser.add_argument(
        '--operation', type=str, default='train',
        choices=('train', 'fine-tune'),
        help='Choose either to train the model or to use a trained one for '
             'detection')
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Path to the directory containing images and labels')
    parser.add_argument(
        '--output_dir', type=str, required=True, default=None,
        help='Path where logs and the model will be saved')
    parser.add_argument(
        '--model', type=str, default='U-Net',
        choices=('U-Net', 'SegNet', 'DeepLab'),
        help='Model architecture')
    parser.add_argument(
        '--model_fn', type=str,
        help='Output model filename')
    parser.add_argument(
        '--weights_path', type=str, default=None,
        help='ONLY FOR OPERATION == FINE-TUNE: Input weights path')
    parser.add_argument(
        '--visualization_path', type=str, default='/tmp',
        help='Path to a directory where the accuracy visualization '
             'will be saved')
    parser.add_argument(
        '--nr_epochs', type=int, default=1,
        help='Number of epochs to train the model. Note that in conjunction '
             'with initial_epoch, epochs is to be understood as the final '
             'epoch')
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
        choices=('binary_crossentropy', 'categorical_crossentropy', 'dice',
                 'tversky'),
        help='A function that maps the training onto a real number '
             'representing cost associated with the epoch')
    parser.add_argument(
        '--seed', type=int, default=1,
        help='Generator random seed')
    parser.add_argument(
        '--patience', type=int, default=100,
        help='Number of epochs with no improvement after which training will '
             'be stopped')
    parser.add_argument(
        '--tensor_height', type=int, default=256,
        help='Height of the tensor representing the image')
    parser.add_argument(
        '--tensor_width', type=int, default=256,
        help='Width of the tensor representing the image')
    parser.add_argument(
        '--monitored_value', type=str, default='val_accuracy',
        help='Metric name to be monitored')
    parser.add_argument(
        '--force_dataset_generation', type=utils.str2bool, default=False,
        help='Boolean to force the dataset structure generation')
    parser.add_argument(
        '--fit_dataset_in_memory', type=utils.str2bool, default=False,
        help='Boolean to load the entire dataset into memory instead '
             'of opening new files with each request - results in the '
             'reduction of I/O operations and time, but could result in huge '
             'memory needs in case of a big dataset')
    parser.add_argument(
        '--augment_training_dataset', type=utils.str2bool, default=False,
        help='Boolean to augment the training dataset with rotations, '
             'shear and flips')
    parser.add_argument(
        '--tversky_alpha', type=float, default=None,
        help='ONLY FOR LOSS_FUNCTION == TVERSKY: Coefficient alpha')
    parser.add_argument(
        '--tversky_beta', type=float, default=None,
        help='ONLY FOR LOSS_FUNCTION == TVERSKY: Coefficient beta')
    parser.add_argument(
        '--dropout_rate_input', type=float, default=None,
        help='Fraction of the input units of the  input layer to drop')
    parser.add_argument(
        '--dropout_rate_hidden', type=float, default=None,
        help='Fraction of the input units of the hidden layers to drop')
    parser.add_argument(
        '--validation_set_percentage', type=float, default=0.2,
        help='If generating the dataset - Percentage of the entire dataset to '
             'be used for the validation or detection in the form of '
             'a decimal number')
    parser.add_argument(
        '--filter_by_classes', type=str, default=None,
        help='If generating the dataset - Classes of interest. If specified, '
             'only samples containing at least one of them will be created. '
             'If filtering by multiple classes, specify their values '
             'comma-separated (e.g. "1,2,6" to filter by classes 1, 2 and 6)')

    args = parser.parse_args()

    # check required arguments by individual operations
    if args.operation == 'fine-tune' and args.model_path is None:
        raise parser.error(
            'Argument model_path required for operation == fine-tune')
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
    if not 0 <= args.validation_set_percentage < 1:
        raise parser.error(
            'Argument validation_set_percentage must be greater or equal to '
            '0 and smaller or equal than 1')

    main(args.operation, args.data_dir, args.output_dir,
         args.model, args.model_fn, args.weights_path, args.visualization_path,
         args.nr_epochs, args.initial_epoch, args.batch_size,
         args.loss_function, args.seed, args.patience,
         (args.tensor_height, args.tensor_width), args.monitored_value,
         args.force_dataset_generation, args.fit_dataset_in_memory,
         args.augment_training_dataset, args.tversky_alpha,
         args.tversky_beta, args.dropout_rate_input,
         args.dropout_rate_hidden, args.validation_set_percentage,
         args.filter_by_classes)
