#!/usr/bin/python3

import os
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from osgeo import gdal
from tensorflow.math import confusion_matrix


def onehot_decode(onehot, colormap, nr_bands=3, enhance_colours=True):
    """Decode onehot mask labels to an eye-readable image.

    :param onehot: one hot encoded image matrix (height x width x
        num_classes)
    :param colormap: dictionary mapping label ids to their codes
    :param nr_bands: number of bands of intended input images
    :param enhance_colours: Enhance the contrast between colours
        (pseudorandom multiplication of the colour value)
    :return: decoded RGB image (height x width x 3)
    """
    # create 2D matrix with label ids (so you do not have to loop)
    single_layer = np.argmax(onehot, axis=-1)

    # create colourful visualizations
    out_shape = (onehot.shape[0], onehot.shape[1], nr_bands)
    output = np.zeros(out_shape)
    for k in colormap.keys():
        output[single_layer == k] = colormap[k]

    if enhance_colours is True:
        multiply_vector = [i ** 3 for i in range(1, nr_bands + 1)]
        enhancement_matrix = np.ones(out_shape) * np.array(multiply_vector,
                                                           dtype=np.uint8)
        output *= enhancement_matrix

    return np.uint8(output)


def write_stats(result, out_path='/tmp/accu.png'):
    """Write graphs with loss, val_loss, accuracy and val_accuracy.

    :param result: output from model.fit()
    :param out_path: a filepath where the graphs will be written into
    """
    # Get actual number of epochs model was trained for
    epochs = len(result.history['loss'])
    epochs_range = np.arange(0, epochs)

    # Plot the model evaluation history
    plt.style.use("ggplot")
    fig = plt.figure(figsize=(40, 16))

    fig.add_subplot(1, 2, 1)
    plt.title("Training Loss")
    plt.plot(epochs_range, result.history["loss"], label="train_loss")
    plt.plot(epochs_range, result.history["val_loss"], label="val_loss")
    plt.ylim(0, 1)

    fig.add_subplot(1, 2, 2)
    plt.title("Training Accuracy")
    plt.plot(epochs_range, result.history["accuracy"],
             label="train_accuracy")
    plt.plot(epochs_range, result.history["val_accuracy"],
             label="val_accuracy")
    plt.ylim(0, 1)

    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    if not os.path.isdir(os.path.split(out_path)[0]):
        os.makedirs(os.path.split(out_path)[0])
    plt.savefig(out_path)

    plt.close()


def visualize_detections(images, ground_truths, detections, id2code,
                         label_codes, label_names, geoinfos, out_dir='/tmp'):
    """Create visualizations.

    Consist of the original image, the confusion matrix, ground truth labels
    and the model predicted labels.

    :param images: original images
    :param ground_truths: ground truth labels
    :param detections: the model label predictions
    :param id2code: dictionary mapping label ids to their codes
    :param label_codes: list with label codes
    :param label_names: list with label names
    :param geoinfos: list in format(filename, projection, geo_transform)
    :param out_dir: directory where the output visualizations will be saved
    """
    max_id = max(id2code.values())
    name_range = range(len(label_names))

    driver = gdal.GetDriverByName("GTiff")
    plt.rcParams['figure.dpi'] = 300

    for i in range(0, np.shape(detections)[0]):
        if i == len(geoinfos):
            # the sample count is not dividable by batch_size
            break

        # THE OVERVIEW IMAGE SECTION

        fig = plt.figure(figsize=(17, 17))

        # original image
        ax1 = fig.add_subplot(2, 2, 1)
        # TODO: expect also other data than S2
        a = np.stack((images[i][:, :, 3], images[i][:, :, 2],
                      images[i][:, :, 1]), axis=2)
        ax1.imshow((255 / a.max() * a).astype(np.uint8))
        ax1.title.set_text('Actual image')

        # ground truths
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_title('Ground truth labels')
        gt_labels = ground_truths[i]
        gt_labels = onehot_decode(gt_labels, id2code)
        ax3.imshow(gt_labels * 4)

        # detections
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_title('Predicted labels')
        detection_decoded = onehot_decode(detections[i], id2code)
        pred_labels = detection_decoded
        ax4.imshow(pred_labels * 4)

        # confusion matrix
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_title('Confusion matrix')
        conf_matrix = confusion_matrix(
            gt_labels[:, :, 0].flatten(), pred_labels[:, :, 0].flatten(),
            max_id + 1)
        # subset to existing classes
        conf_matrix = conf_matrix.numpy()[label_codes][:, label_codes]
        # normalize the confusion matrix
        row_sums = conf_matrix.sum(axis=1)[:, np.newaxis]
        # TODO: solve division by 0
        cm_norm = np.around(conf_matrix.astype('float') / row_sums, decimals=2)
        # visualize
        ax2.imshow(cm_norm, cmap=plt.cm.Blues)
        y_labels = ['{}\n{}'.format(label_names[j], row_sums[j]) for j in
                    name_range]
        plt.xticks(name_range, label_names)
        plt.yticks(name_range, y_labels)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        # write percentage values (0.00 -- 1.00) into the confusion matrix
        threshold = cm_norm.max() / 2.  # used to decide for the font colour
        for row in range(len(conf_matrix)):
            for col in range(len(conf_matrix)):
                if cm_norm[col, row] > threshold:
                    colour = 'white'
                else:
                    colour = 'black'
                # TODO: class names, not codes
                ax2.text(row, col, cm_norm[col, row], color=colour,
                         horizontalalignment='center')

        # save the overview image
        plt.savefig(os.path.join(out_dir, geoinfos[i][0][:-4]))
        plt.close()

        # THE DETECTION TIF IMAGE SECTION

        out = driver.Create(os.path.join(out_dir, f'{geoinfos[i][0]}'),
                            np.shape(detections)[2],
                            np.shape(detections)[1],
                            1,
                            gdal.GDT_Byte)
        outband = out.GetRasterBand(1)
        outband.WriteArray(detection_decoded[:, :, 0], 0, 0)
        out.SetProjection(geoinfos[i][1])
        out.SetGeoTransform(geoinfos[i][2])

        out = None
