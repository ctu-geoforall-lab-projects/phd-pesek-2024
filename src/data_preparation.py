#!/usr/bin/python3

import os
import glob
import shutil

import numpy as np
import tensorflow as tf

from osgeo import gdal


def read_images(data_dir, tensor_shape=(256, 256), verbose=1):
    """Read images and return them as tensors and lists of filenames.

    :param data_dir: path to the directory containing images
    :param tensor_shape: shape of the first two dimensions of input tensors
    :param verbose: verbosity (0=quiet, >0 verbose)
    :return: image_tensors, masks_tensors
    """
    images_arrays = []
    masks_arrays = []
    for i in glob.glob(os.path.join(data_dir, '*image.tif')):
        tiled = tile(i, i.replace('image.tif', 'label.tif'), tensor_shape)
        images_arrays.extend(tiled[0])
        masks_arrays.extend(tiled[1])

    if masks_arrays[0].ndim == 2:
        masks_arrays = [np.expand_dims(i, -1) for i in masks_arrays]

    # create TF datasets
    images_dataset = tf.data.Dataset.from_tensor_slices(images_arrays)
    masks_dataset = tf.data.Dataset.from_tensor_slices(masks_arrays)

    im_nr = len(images_arrays)
    if verbose > 0:
        print('Created {} training samples from the provided '
              'image.'.format(im_nr))

    return images_dataset, masks_dataset


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
    # function to be used while saving samples
    def train_val_determination(val_set_pct):
        """Return decision about the sample will be part of train or val set."""
        pct = 0
        while True:
            pct += val_set_pct
            print(pct)
            if pct < 1:
                yield 'train'
            else:
                pct -= 1
                yield 'val'

    # Create folders to hold images and masks
    dirs = ['train_images', 'train_masks', 'val_images', 'val_masks']

    for directory in dirs:
        dir_full_path = os.path.join(data_dir, directory)
        if os.path.isdir(dir_full_path):
            shutil.rmtree(dir_full_path)

        os.makedirs(dir_full_path)

    images, masks = read_images(data_dir, tensor_shape)

    # TODO: would be nice to avoid tf.compat.v1 (stay v2) (what about my
    #       generator?)
    # Create iterators for images and masks
    # outside of TF Eager, we would use make_one_shot_iterator
    frame_batches = tf.compat.v1.data.make_one_shot_iterator(images)
    mask_batches = tf.compat.v1.data.make_one_shot_iterator(masks)

    driver = gdal.GetDriverByName('GTiff')

    # Iterate over the images while saving the images and masks
    # in appropriate folders
    im_id = 0
    dir_names = train_val_determination(val_set_pct)
    for image, mask in zip(frame_batches, mask_batches):
        # TODO: Experiment with uint16
        # Convert tensors to numpy arrays
        image = (image.numpy() / 255).astype(np.uint8)
        mask = mask.numpy().astype(np.uint8)

        # TODO: Avoid two transpositions
        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        # TODO: https://stackoverflow.com/questions/53776506/how-to-save-an-array-representing-an-image-with-40-band-to-a-tif-file

        dir_name = next(dir_names)
        image_path = os.path.join(data_dir,
                                  '{}_images'.format(dir_name),
                                  'image_{0:03d}.tif'.format(im_id + 1))
        mask_path = os.path.join(data_dir,
                                 '{}_masks'.format(dir_name),
                                 'image_{0:03d}.tif'.format(im_id + 1))

        # write rasters
        dout = driver.Create(image_path, tensor_shape[0],
                             tensor_shape[1], nr_bands, gdal.GDT_UInt16)
        for i in range(nr_bands):
            dout.GetRasterBand(i + 1).WriteArray(image[i])

        dout = driver.Create(mask_path, tensor_shape[0],
                             tensor_shape[1], 1, gdal.GDT_UInt16)
        for i in range(1):
            dout.GetRasterBand(i + 1).WriteArray(mask[i])

        im_id += 1

    if verbose > 0:
        print("Saved {} images to directory {}".format(im_id, data_dir))


def tile(scene_path, labels_path, tensor_shape):
    import pyjeo as pj

    scene_nps = []
    labels_nps = []

    # load images
    scene = pj.Jim(scene_path)
    labels = pj.Jim(labels_path)

    nr_col = scene.properties.nrOfCol()
    nr_row = scene.properties.nrOfRow()
    cols_step = tensor_shape[0]
    rows_step = tensor_shape[1]

    for i in range(0, nr_col, cols_step):
        for j in range(0, nr_row, rows_step):
            # if reaching the end of the image, expand the window back to
            # avoid pixels outside the image
            if j + rows_step > nr_row:
                j = nr_row - rows_step
            if i + cols_step > nr_col:
                i = nr_col - cols_step

            # crop images
            scene_cropped = pj.geometry.crop(scene, ulx=i, uly=j,
                                             lrx=i + cols_step,
                                             lry=j + rows_step,
                                             nogeo=True)
            labels_cropped = pj.geometry.crop(labels, ulx=i, uly=j,
                                              lrx=i + cols_step,
                                              lry=j + rows_step,
                                              nogeo=True)

            # stack bands
            scene_np = np.stack([scene_cropped.np(i) for i in
                                 range(scene_cropped.properties.nrOfBand())],
                                axis=2)
            labels_np = labels_cropped.np()

            scene_nps.append(scene_np)
            labels_nps.append(labels_np)

    return scene_nps, labels_nps
