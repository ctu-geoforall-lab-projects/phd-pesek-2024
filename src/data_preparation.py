#!/usr/bin/python3

import os
import shutil

import numpy as np
import tensorflow as tf

from osgeo import gdal


def read_images(data_dir, tensor_shape=(256, 256), verbose=1):
    """Read images and return them as tensors and lists of filenames.

    :param data_dir: path to the directory containing images
    :param tensor_shape: shape of the first two dimensions of input tensors
    :param verbose: verbosity (0=quiet, >0 verbose)
    :return: image_tensors, masks_tensors, images_filenames, masks_filenames
    """
    # Get the file names list from provided directory
    file_list = [f for f in os.listdir(data_dir) if
                 os.path.isfile(os.path.join(data_dir, f))]
    file_list = sorted(file_list)

    # when sorted, images 1 to 10 are not followed by their labels
    for i in [str(j) for j in range(1, 11)]:
        string = 'image_{}.tif'.format(i)
        string_L = 'image_{}_L.tif'.format(i)
        file_list.remove(string_L)
        file_list.insert(file_list.index(string) + 1, string_L)

    # Separate frame and mask files lists, exclude unnecessary files
    images_filenames = [file for file in file_list if
                        ('_L' not in file) and ('txt' not in file)]
    masks_filenames = [file for file in file_list if
                       ('_L' in file) and ('txt' not in file)]

    if verbose > 0:
        print('{} image files found in the provided directory.'.format(
            len(images_filenames)))
        print('{} mask files found in the provided directory.'.format(
            len(masks_filenames)))

    # Create file paths from file names
    images_paths = [os.path.join(data_dir, fname) for fname in images_filenames]
    masks_paths = [os.path.join(data_dir, fname) for fname in masks_filenames]

    # Create dataset of np arrays
    images_arrays = [
        gdal.Open(i, gdal.GA_ReadOnly).ReadAsArray() for i in images_paths]
    images_arrays = [np.transpose(i, (1, 2, 0)) for i in images_arrays]
    masks_arrays = [
        gdal.Open(i, gdal.GA_ReadOnly).ReadAsArray() for i in masks_paths]
    if masks_arrays[0].ndim == 2:
        masks_arrays = [np.expand_dims(i, -1) for i in masks_arrays]

    # resize tensors to the specified shape
    image_tensors = map(lambda x: tf.image.resize(x, tensor_shape),
                        images_arrays)
    mask_tensors = map(lambda x: tf.image.resize(x, tensor_shape),
                       masks_arrays)

    # create TF datasets
    images_dataset = tf.data.Dataset.from_tensor_slices(list(image_tensors))
    masks_dataset = tf.data.Dataset.from_tensor_slices(list(mask_tensors))

    if verbose > 0:
        print('Completed importing {} images from the provided '
              'directory.'.format(len(images_filenames)))
        print('Completed importing {} masks from the provided '
              'directory.'.format(len(masks_filenames)))

    return images_dataset, masks_dataset, images_filenames, masks_filenames


def parse_label_code(line):
    """Parse lines in a text file into a label code and a label name.

    :param line: line in the txt file
    :return: tuple with an integer label code, a string label name
    """
    a, b = line.strip().split(',')

    # format label_value, label_name
    return int(a), b


def generate_dataset_structure(data_dir, nr_bands=12, tensor_shape=(256, 256),
                               val_set_pct=0.2, verbose=1):
    """Generate the expected dataset structure.

    Will generate directories train_images, train_masks, val_images and
    val_masks.

    :param data_dir: path to the directory containing images
    :param nr_bands: number of bands of intended input images
    :param tensor_shape: shape of the first two dimensions of input tensors
    :param val_set_pct: percentage of the validation images in the dataset
    :param verbose: verbosity (0=quiet, >0 verbose)
    """
    # Create folders to hold images and masks
    dirs = ['train_images', 'train_masks', 'val_images', 'val_masks']

    for directory in dirs:
        dir_full_path = os.path.join(data_dir, directory)
        if os.path.isdir(dir_full_path):
            shutil.rmtree(dir_full_path)

        os.makedirs(dir_full_path)

    images, masks, images_filenames, masks_filenames = read_images(
        data_dir, tensor_shape)

    # TODO: would be nice to avoid tf.compat.v1 (stay v2) (what about my
    #       generator?)
    # Create iterators for images and masks
    # outside of TF Eager, we would use make_one_shot_iterator
    frame_batches = tf.compat.v1.data.make_one_shot_iterator(images)
    mask_batches = tf.compat.v1.data.make_one_shot_iterator(masks)

    driver = gdal.GetDriverByName('GTiff')


    # create mappings with corresponding dirs (train/val)
    val_im_nr = round(val_set_pct * len(images_filenames))
    corresponding_dirs = ('train',) * (len(images_filenames) - val_im_nr)
    corresponding_dirs += ('val',) * val_im_nr

    # Iterate over the images while saving the images and masks
    # in appropriate folders
    files = zip(images_filenames, corresponding_dirs)
    for file, dir_name in files:
        # TODO: Experiment with uint16
        # Convert tensors to numpy arrays
        image = (frame_batches.next().numpy() / 255).astype(np.uint8)
        mask = mask_batches.next().numpy().astype(np.uint8)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        # TODO: https://stackoverflow.com/questions/53776506/how-to-save-an-array-representing-an-image-with-40-band-to-a-tif-file

        image_path = os.path.join(data_dir, '{}_images'.format(dir_name),
                                  file)
        mask_path = os.path.join(data_dir, '{}_masks'.format(dir_name),
                                 file)

        # write rasters
        dout = driver.Create(image_path, tensor_shape[0],
                             tensor_shape[1], nr_bands, gdal.GDT_UInt16)
        for i in range(nr_bands):
            dout.GetRasterBand(i + 1).WriteArray(image[i])

        dout = driver.Create(mask_path, tensor_shape[0],
                             tensor_shape[1], 1, gdal.GDT_UInt16)
        for i in range(1):
            dout.GetRasterBand(i + 1).WriteArray(mask[i])

    if verbose > 0:
        print("Saved {} images to directory {}".format(
            len(images_filenames), data_dir))
        print("Saved {} masks to directory {}".format(
            len(masks_filenames), data_dir))
