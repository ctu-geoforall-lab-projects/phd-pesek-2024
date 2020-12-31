#!/usr/bin/python3

import os
import glob
import argparse

import numpy as np
import tensorflow as tf

from osgeo import gdal
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, \
    EarlyStopping

from data_preparation import parse_label_code, generate_dataset_structure
from cnn_lib import TrainAugmentGenerator, ValAugmentGenerator
from architectures import get_unet
from visualization import write_stats, visualize_detections


def main(operation, data_dir, output_dir, model_fn, in_model_path,
         visualization_path, nr_epochs, batch_size, seed, patience,
         tensor_shape, monitored_value):
    print_device_info()

    # get nr of bands
    dataset = glob.glob(os.path.join(data_dir, '*[0-9].tif'))
    dataset_image = gdal.Open(dataset[0])
    nr_bands = dataset_image.RasterCount
    dataset_image = None

    generate_dataset_structure(data_dir, nr_bands, tensor_shape)

    label_codes, label_names, id2code = get_codings(
        os.path.join(data_dir, 'label_colors.txt'))

    # set TensorFlow seed
    tf.random.set_seed(seed)

    model = create_model(len(id2code), nr_bands, tensor_shape)

    # TODO: read nr of samples automatically
    if operation == 'train':
        # Train model
        train(data_dir, model, id2code, batch_size, output_dir,
              visualization_path, model_fn, nr_epochs, 100, seed=seed,
              patience=patience, monitored_value=monitored_value)
    else:
        # detect
        detect(data_dir, model, in_model_path, id2code, batch_size,
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
    # TODO: do I need code2id, id2name and name2id?
    code2id = {j: i for i, j in enumerate(label_codes)}
    id2code = {i: j for i, j in enumerate(label_codes)}
    name2id = {j: i for i, j in enumerate(label_names)}
    id2name = {i: j for i, j in enumerate(label_names)}

    return label_codes, label_names, id2code


def create_model(nr_classes, nr_bands, tensor_shape, optimizer='adam',
                 loss='categorical_crossentropy', metrics=None, verbose=1):
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
    :param verbose: verbosity (0=quiet, >0 verbose)
    :return: compiled model
    """
    if metrics is None:
        metrics = ['accuracy']

    model = get_unet(nr_classes, nr_bands=nr_bands, nr_filters=32,
                     tensor_shape=tensor_shape)

    # TODO: check other metrics (tversky loss, dice coef)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    if verbose > 0:
        model.summary()

    return model


# TODO: support initial_epoch for fine-tuning
def train(data_dir, model, id2code, batch_size, output_dir,
          visualization_path, model_fn, nr_epochs, nr_samples, seed=1,
          patience=100, monitored_value='val_accuracy'):
    """Run model training.

    :param data_dir: path to the directory containing images
    :param model: model to be used for the detection
    :param id2code: dictionary mapping label ids to their codes
    :param batch_size: the number of samples that will be propagated through
        the network at once
    :param output_dir: path where logs and the model will be saved
    :param visualization_path: path to a directory where the output
        visualizations will be saved
    :param model_fn: model file name
    :param nr_epochs: number of epochs to train the model
    :param nr_samples: sum of training and validation samples together
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

    # TODO: check because of generators
    # TODO: parameterize?
    steps_per_epoch = np.ceil(float(
        nr_samples - round(0.1 * nr_samples)) / float(batch_size))
    validation_steps = (
        float((round(0.1 * nr_samples))) / float(batch_size))

    # train
    # TODO: check fit_generator()
    result = model.fit(
        TrainAugmentGenerator()(data_dir, id2code, seed, batch_size),
        validation_data=ValAugmentGenerator()(data_dir, id2code, seed,
                                              batch_size),
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=nr_epochs,
        callbacks=callbacks)
    # TODO: is it needed with the model checkpoint?
    model.save_weights(out_model_path, overwrite=True)

    write_stats(result, os.path.join(visualization_path, 'accu.png'))


def detect(data_dir, model, in_model_path, id2code, batch_size, label_codes,
           label_names, seed=1, out_dir='/tmp'):
    """Run detection.

    :param data_dir: path to the directory containing images
    :param model: model to be used for the detection
    :param in_model_path: path to a model to be loaded for the detection
    :param id2code: dictionary mapping label ids to their codes
    :param batch_size: the number of samples that will be propagated through
        the network at once
    :param label_codes: list with label codes
    :param label_names: list with label names
    :param seed: the generator seed
    :param out_dir: directory where the output visualizations will be saved
    """
    # TODO: Do not test on augmented data
    testing_gen = ValAugmentGenerator(data_dir, id2code, seed, batch_size)

    batch_img, batch_mask = next(testing_gen)
    model.load_weights(in_model_path)
    pred_all = model.predict(batch_img)

    visualize_detections(batch_img, batch_mask, pred_all, id2code,
                         label_codes, label_names, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run U-Net training and detection')

    parser.add_argument(
        '--operation', type=str, required=True, choices=('train', 'detect'),
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
        help='ONLY FOR OPERATION == DETECT: Input model path')
    parser.add_argument(
        '--visualization_path', type=str, default='/tmp',
        help='Path to a directory where the accuracy visualization '
             '(operation == train) or detection visualizations '
             '(operation == detect) will be saved')
    parser.add_argument(
        '--nr_epochs', type=int, default=1,
        help='ONLY FOR OPERATION == TRAIN: Number of epochs to train '
             'the model')
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='The number of samples that will be propagated through the '
             'network at once')
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

    args = parser.parse_args()

    # check required arguments by individual operations
    if args.operation == 'train' and args.output_dir is None:
        raise parser.error(
            'Argument output_dir required for operation == train')
    if args.operation == 'detect' and args.model_path is None:
        raise parser.error(
            'Argument model_path required for operation == detect')

    main(args.operation, args.data_dir, args.output_dir, args.model_fn,
         args.model_path, args.visualization_path, args.nr_epochs,
         args.batch_size, args.seed, args.patience,
         (args.tensor_height, args.tensor_width), args.monitored_value)
