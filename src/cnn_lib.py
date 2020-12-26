#!/usr/bin/python3

import os
import rasterio

import numpy as np


# TODO: check keras.utils.to_categorical
def onehot_encode(orig_image, colormap):
    """Encode input images into one hot ones.

    :param orig_image: original image
    :param colormap: dictionary mapping label ids to their codes
    :return: One hot encoded image of dimensions (height x width x num_classes)
    """
    num_classes = len(colormap)
    shape = orig_image.shape[:2] + (num_classes,)
    # TODO: Test with np.empty, np.uint8
    encoded_image = np.zeros(shape, dtype=np.int8)
    for i, cls in enumerate(colormap):
        tup = (-1, 1)
        resh = orig_image.reshape(tup)
        eq = resh == colormap[i]
        all_ax = np.all(eq, axis=1)
        encoded_image[:, :, i] = all_ax.reshape(shape[:2])

    return encoded_image


# TODO: get rid of the rasterio dependency
def rasterio_generator(data_dir, rescale=False, batch_size=5):
    """Generate batches of images.

    :param data_dir: path to the directory containing images
    :param rescale: boolean saying whether to rescale images or not
        (rescaling is a division by 255)
    :param batch_size: the number of samples that will be propagated through
        the network at once
    :return: yielded batch-sized np stack of images
    """
    index = 1
    batch = []
    while True:
        for file in sorted(os.listdir(data_dir)):
            a = rasterio.open(os.path.join(data_dir, file))
            q = a.read()
            q = np.transpose(q, (1, 2, 0))
            if rescale:
                q = 1. / 255 * q

            batch.append(q)

            if index % batch_size == 0:
                yield np.stack(batch)
                batch = []

            index += 1


# TODO: check tf.data.Dataset.from_generator()
# TODO: check tf.keras.preprocessing.image.ImageDataGenerator
# TODO: check keras.utils.Sequence
# TODO: Does not really augment, does it?
# TODO: support onehot_encode boolean parameter
# TODO: should be a class
def TrainAugmentGenerator(data_dir, id2code, seed=1, batch_size=5):
    """Generate batches of training data.

    :param data_dir: path to the directory containing images
    :param id2code: dictionary mapping label ids to their codes
    :param seed: the generator seed
    :param batch_size: the number of samples that will be propagated through
        the network at once
    :return: yielded tuple of batch-sized np stacks of training images and
        masks
    """
    # TODO: the train directory seems redundant
    train_image_generator = rasterio_generator(
        os.path.join(data_dir, 'train_images', 'train'), False, batch_size)
    train_mask_generator = rasterio_generator(
        os.path.join(data_dir, 'train_masks', 'train'), False, batch_size)

    while True:
        x1i = next(train_image_generator)
        x2i = next(train_mask_generator)

        # TODO: have seen the following somewhere - check
        # One hot encoding RGB images
        # mask_encoded = [onehot_encode(x2i[0][x, :, :, :], id2code) for x in
        #                 range(x2i[0].shape[0])]
        # yield x1i[0], np.asarray(mask_encoded)

        # one hot encode masks
        mask_encoded = [onehot_encode(x2i[x, :, :, :], id2code) for x in
                        range(x2i.shape[0])]

        yield x1i, np.asarray(mask_encoded)


# TODO: can probably be squashed with the TrainAugmentGenerator
# TODO: check tf.data.Dataset.from_generator()
# TODO: check tf.keras.preprocessing.image.ImageDataGenerator
# TODO: check keras.utils.Sequence
# TODO: Does not really augment, does it?
# TODO: support onehot_encode boolean parameter
# TODO: should be a class
def ValAugmentGenerator(data_dir, id2code, seed=1, batch_size=5):
    """Generate batches of validation data.

    :param data_dir: path to the directory containing images
    :param id2code: dictionary mapping label ids to their codes
    :param seed: the generator seed
    :param batch_size: the number of samples that will be propagated through
        the network at once
    :return: yielded tuple of batch-sized np stacks of validation images and
        masks
    """
    # TODO: the val directory seems redundant
    val_image_generator = rasterio_generator(
        os.path.join(data_dir, 'val_images', 'val'), False, batch_size)
    val_mask_generator = rasterio_generator(
        os.path.join(data_dir, 'val_masks', 'val'), False, batch_size)

    while True:
        x1i = next(val_image_generator)
        x2i = next(val_mask_generator)

        # TODO: has seen the following somewhere - check
        # mask_encoded = [onehot_encode(x2i[0][x, :, :, :], id2code) for x in
        #                 range(x2i[0].shape[0])]

        # one hot encode masks
        mask_encoded = [onehot_encode(x2i[x, :, :, :], id2code) for x in
                        range(x2i.shape[0])]

        yield x1i, np.asarray(mask_encoded)
