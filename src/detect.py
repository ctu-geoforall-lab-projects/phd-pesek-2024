#!/usr/bin/python3

import os
import argparse

import tensorflow as tf

from osgeo import gdal

# imports from this package
import utils

from cnn_lib import AugmentGenerator
from architectures import create_model
from visualization import visualize_detections


def main(data_dir, model, in_weights_path, visualization_path, batch_size,
         seed, tensor_shape, force_dataset_generation, fit_memory, val_set_pct,
         filter_by_class):
    utils.print_device_info()

    # get nr of bands
    nr_bands = utils.get_nr_of_bands(data_dir)

    label_codes, label_names, id2code = utils.get_codings(
        os.path.join(data_dir, 'label_colors.txt'))

    # set TensorFlow seed
    tf.random.set_seed(seed)

    model = create_model(model, len(id2code), nr_bands, tensor_shape)

    # val generator used for both the training and the detection
    val_generator = AugmentGenerator(
        data_dir, batch_size, 'val', tensor_shape, force_dataset_generation,
        fit_memory, val_set_pct=val_set_pct, filter_by_class=filter_by_class)

    # load weights if the model is supposed to do so
    model.load_weights(in_weights_path)
    model.set_weights(utils.model_replace_nans(model.get_weights()))

    detect(model, val_generator, id2code, [i for i in label_codes],
           label_names, data_dir, seed, visualization_path)


def detect(model, val_generator, id2code, label_codes, label_names,
           data_dir, seed=1, out_dir='/tmp'):
    """Run detection.

    :param model: model to be used for the detection
    :param val_generator: validation data generator
    :param id2code: dictionary mapping label ids to their codes
    :param label_codes: list with label codes
    :param label_names: list with label names
    :param data_dir: path to the directory containing images and labels
    :param seed: the generator seed
    :param out_dir: directory where the output visualizations will be saved
    """
    testing_gen = val_generator(id2code, seed)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # get information needed to write referenced geotifs of detections
    geoinfos = get_geoinfo(val_generator.masks_dir)

    batch_size = val_generator.batch_size

    for i in range(0, val_generator.nr_samples, batch_size):
        # get a batch of data
        batch_img, batch_mask = next(testing_gen)
        pred_all = model.predict(batch_img)

        batch_geoinfos = geoinfos[i:i + batch_size]

        # visualize the natch
        visualize_detections(batch_img, batch_mask, pred_all, id2code,
                             label_codes, label_names, batch_geoinfos,
                             out_dir)


def get_geoinfo(data_dir):
    """Get information needed to write referenced geotifs of detections.

    :param data_dir: path to the directory with either val_images
        or val_masks
    :return: list of sets in format [(filenames, projs, geo_transforms), ...]
    """
    filenames = []
    projs = []
    geo_transforms = []

    for filename in sorted(os.listdir(data_dir)):
        src = gdal.Open(os.path.join(data_dir, filename), gdal.GA_ReadOnly)
        filenames.append(filename)
        projs.append(src.GetProjection())
        geo_transforms.append(src.GetGeoTransform())

        src = None

    return [i for i in zip(filenames, projs, geo_transforms)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run detection')

    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Path to the directory containing images and labels')
    parser.add_argument(
        '--model', type=str, default='U-Net',
        choices=('U-Net', 'SegNet', 'DeepLab'),
        help='Model architecture')
    parser.add_argument(
        '--weights_path', type=str, default=None,
        help='Input weights path')
    parser.add_argument(
        '--visualization_path', type=str, default='/tmp',
        help='Path to a directory where the detection visualizations '
             'will be saved')
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='The number of samples that will be propagated through the '
             'network at once')
    parser.add_argument(
        '--seed', type=int, default=1,
        help='Generator random seed')
    parser.add_argument(
        '--tensor_height', type=int, default=256,
        help='Height of the tensor representing the image')
    parser.add_argument(
        '--tensor_width', type=int, default=256,
        help='Width of the tensor representing the image')
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
    if args.weights_path is None:
        raise parser.error(
            'Argument weights_path required')
    if not 0 <= args.validation_set_percentage <= 1:
        raise parser.error(
            'Argument validation_set_percentage must be greater or equal to 0 '
            'and smaller than 1')

    main(args.data_dir, args.model, args.weights_path, args.visualization_path,
         args.batch_size, args.seed, (args.tensor_height, args.tensor_width),
         args.force_dataset_generation, args.fit_dataset_in_memory,
         args.validation_set_percentage, args.filter_by_classes)
