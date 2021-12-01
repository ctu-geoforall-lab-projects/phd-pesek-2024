#!/usr/bin/python3

import os
import glob
import argparse

import tensorflow as tf

from osgeo import gdal


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


def get_nr_of_bands(data_dir):
    images = glob.glob(os.path.join(data_dir, '*image.tif'))
    dataset_image = gdal.Open(images[0], gdal.GA_ReadOnly)
    nr_bands = dataset_image.RasterCount
    dataset_image = None

    return nr_bands


def parse_label_code(line):
    """Parse lines in a text file into a label code and a label name.

    :param line: line in the txt file
    :return: tuple with an integer label code, a string label name
    """
    a, b = line.strip().split(',')

    # format label_value, label_name
    return int(a), b


def print_device_info():
    """Print info about used GPUs."""
    print('Available GPUs:')
    print(tf.config.list_physical_devices('GPU'))

    print('Device name:')
    print(tf.random.uniform((1, 1)).device)

    print('TF executing eagerly:')
    print(tf.executing_eagerly())


def str2bool(string_val):
    """Transform a string looking like a boolean value to a boolean value.

    This is needed because using type=bool in argparse actually parses strings.
    Such an behaviour could result in `--force_dataset_generation False` being
    misinterpreted as True (bool('False') == True).

    :param string_val: a string looking like a boolean value
    :return: the corresponding boolean value
    """
    if isinstance(string_val, bool):
        return string_val
    elif string_val.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif string_val.lower() in ('false', 'no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
