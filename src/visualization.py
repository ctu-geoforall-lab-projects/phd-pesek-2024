#!/usr/bin/python3

import os
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.math import confusion_matrix


def onehot_decode(onehot, colormap, nr_bands=3, enhance_colours=True):
    """Decode onehot mask labels to an eye-readable image.

    :param onehot: one hot encoded image matrix (height x width x
        num_classes)
    :param colormap: dictionary mapping label ids to their codes
    :param nr_bands: number of bands of intended input images
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


# TODO: parameter smooth
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
    plt.savefig(out_path)

    plt.close()


def visualize_detections(images, ground_truths, detections, id2code,
                         label_codes, label_names, out_dir='/tmp'):
    """Create visualizations.

    Consist of the original image, the confusion matrix, ground truth labels
    and the model predicted labels.

    :param images: original images
    :param ground_truths: ground truth labels
    :param detections: the model label predictions
    :param id2code: dictionary mapping label ids to their codes
    :param label_codes: list with label codes
    :param label_names: list with label names
    :param out_dir: directory where the output visualizations will be saved
    """
    max_id = max(id2code.values())[0]
    name_range = range(len(label_names))

    for i in range(0, np.shape(detections)[0]):
        fig = plt.figure(figsize=(17, 17))

        # original image
        ax1 = fig.add_subplot(2, 2, 1)
        # TODO: expect also other data than S2
        a = np.stack((images[i][:, :, 3], images[i][:, :, 2],
                      images[i][:, :, 1]), axis=2)
        ax1.imshow((255 / a.max() * a).astype(np.uint8))
        ax1.title.set_text('Actual image')
        ax1.grid(b=None)

        # ground truths
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_title('Ground truth labels')
        gt_labels = ground_truths[i]
        gt_labels = onehot_decode(gt_labels, id2code)
        ax3.imshow(gt_labels)
        ax3.grid(b=None)

        # detections
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_title('Predicted labels')
        pred_labels = onehot_decode(detections[i], id2code)
        ax4.imshow(pred_labels)
        ax4.grid(b=None)

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
        y_labels = ['{}\n{}'.format(label_names[i], row_sums[i]) for i in
                    name_range]
        plt.xticks(name_range, label_names)
        plt.yticks(name_range, y_labels)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        ax2.grid(b=None)
        # write values into the confusion number
        threshold = cm_norm.max() / 2.
        for row in range(len(conf_matrix)):
            for col in range(len(conf_matrix)):
                if cm_norm[col, row] > threshold:
                    colour = 'white'
                else:
                    colour = 'black'
                ax2.text(row, col, cm_norm[col, row], color=colour,
                         horizontalalignment='center')

        plt.savefig(os.path.join(out_dir, str(i)))
        plt.close()
