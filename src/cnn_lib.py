#!/usr/bin/python3

import os
import rasterio

import numpy as np

from data_preparation import generate_dataset_structure


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
class AugmentGenerator:
    """Data generator."""

    def __init__(self, data_dir, batch_size=5, operation='train',
                 nr_bands=12, tensor_shape=(256, 256),
                 force_dataset_generation=False):
        """

        :param data_dir: path to the directory containing images
        :param batch_size: the number of samples that will be propagated
            through the network at once
        :param operation: either 'train' or 'val'
        :param nr_bands: number of bands of intended input images
        :param tensor_shape: shape of the first two dimensions of input tensors
        :param force_dataset_generation: boolean to force the dataset
            structure generation
        """
        if operation not in ('train', 'val'):
            raise AttributeError('Only values "train" and "val" supported as '
                                 'operation. "{}" was given'.format(operation))

        # TODO: the operation directory seems redundant
        images_dir = os.path.join(
            data_dir, '{}_images'.format(operation), operation)
        masks_dir = os.path.join(
            data_dir, '{}_masks'.format(operation), operation)
        # generate the dataset structure if not generated
        do_exist = [os.path.isdir(i) is True for i in (images_dir, masks_dir)]
        if force_dataset_generation or not all(do_exist):
            generate_dataset_structure(data_dir, nr_bands, tensor_shape)

        # create generators
        self.image_generator = rasterio_generator(
            images_dir, False, batch_size)
        self.mask_generator = rasterio_generator(
            masks_dir, False, batch_size)

    def __call__(self, id2code, seed=1):
        """Generate batches of data.

        :param id2code: dictionary mapping label ids to their codes
        :param seed: the generator seed
        :return: yielded tuple of batch-sized np stacks of validation images and
            masks
        """
        while True:
            x1i = next(self.image_generator)
            x2i = next(self.mask_generator)

            # TODO: has seen the following somewhere - check
            # mask_encoded = [onehot_encode(x2i[0][x, :, :, :], id2code) for x in
            #                 range(x2i[0].shape[0])]

            # one hot encode masks
            mask_encoded = [onehot_encode(x2i[x, :, :, :], id2code) for x in
                            range(x2i.shape[0])]

            yield x1i, np.asarray(mask_encoded)
