# Source code

Here, you can find the code to run the training and to run the
detection/prediction if you already have a trained model (either yours or one
provided by a third party).

## Dataset structure

The dataset is expected to be provided in the following structure:

```
path/to/my/dataset/
├── label_colors.txt
├── im1_image.tif
├── im1_label.tif
├── im2_image.tif
├── im2_label.tif
└── ...
```

`label_colors.txt` contains mappings of classes that can be found in the
`*label.tif` files to the colours we plan to visualize them with. The filename
is strict. An example of how such file could look like for a dataset with three
classes:

```
1,1
2,2
3,64
```

During the training or detection step, four extra folders are going to be
created; they will serve as data sources for the operation. They will contain
patches of a specified size (see parameters of the scripts below) created from
the original images. In the case these directories already exist when running a
script, no data preparation will be done (it is recommended to use the
`--force_dataset_generation` in such case if the source dataset has been
modified).

An example of the dataset structure after the creation of the intermediate
directories.

```
path/to/my/dataset/
├── label_colors.txt
├── im1_image.tif
├── im1_label.tif
├── im2_image.tif
├── im2_label.tif
├── train_images
│   ├── image_0.tif
│   ├── image_1.tif
│   ├── image_2.tif
│   └── image_4.tif
├── train_masks
│   ├── image_0.tif
│   ├── image_1.tif
│   ├── image_2.tif
│   └── image_4.tif
├── val_images
│   └── image_3.tif
└── val_masks
    └── image_3.tif
```

## Training

To train your model, use the script `train.py`. See the `--help` of the script
below:

```
usage: train.py [-h] [--operation {train,fine-tune}] --data_dir DATA_DIR
                --output_dir OUTPUT_DIR [--model {U-Net,SegNet,DeepLab}]
                [--model_fn MODEL_FN] [--weights_path WEIGHTS_PATH]
                [--visualization_path VISUALIZATION_PATH]
                [--nr_epochs NR_EPOCHS] [--initial_epoch INITIAL_EPOCH]
                [--batch_size BATCH_SIZE]
                [--loss_function {binary_crossentropy,categorical_crossentropy,dice,tversky}]
                [--seed SEED] [--patience PATIENCE]
                [--tensor_height TENSOR_HEIGHT] [--tensor_width TENSOR_WIDTH]
                [--monitored_value MONITORED_VALUE]
                [--force_dataset_generation FORCE_DATASET_GENERATION]
                [--fit_dataset_in_memory FIT_DATASET_IN_MEMORY]
                [--augment_training_dataset AUGMENT_TRAINING_DATASET]
                [--tversky_alpha TVERSKY_ALPHA] [--tversky_beta TVERSKY_BETA]
                [--dropout_rate_input DROPOUT_RATE_INPUT]
                [--dropout_rate_hidden DROPOUT_RATE_HIDDEN]
                [--validation_set_percentage VALIDATION_SET_PERCENTAGE]
                [--filter_by_classes FILTER_BY_CLASSES]
                [--backbone {ResNet50,ResNet101,ResNet152}]

Run training or fine-tuning

optional arguments:
  -h, --help            show this help message and exit
  --operation {train,fine-tune}
                        Choose either to train the model or to use a trained
                        one for detection
  --data_dir DATA_DIR   Path to the directory containing images and labels
  --output_dir OUTPUT_DIR
                        Path where logs and the model will be saved
  --model {U-Net,SegNet,DeepLab}
                        Model architecture
  --model_fn MODEL_FN   Output model filename
  --weights_path WEIGHTS_PATH
                        ONLY FOR OPERATION == FINE-TUNE: Input weights path
  --visualization_path VISUALIZATION_PATH
                        Path to a directory where the accuracy visualization
                        will be saved
  --nr_epochs NR_EPOCHS
                        Number of epochs to train the model. Note that in
                        conjunction with initial_epoch, epochs is to be
                        understood as the final epoch
  --initial_epoch INITIAL_EPOCH
                        ONLY FOR OPERATION == FINE-TUNE: Epoch at which to
                        start training (useful for resuming a previous
                        training run)
  --batch_size BATCH_SIZE
                        The number of samples that will be propagated through
                        the network at once
  --loss_function {binary_crossentropy,categorical_crossentropy,dice,tversky}
                        A function that maps the training onto a real number
                        representing cost associated with the epoch
  --seed SEED           Generator random seed
  --patience PATIENCE   Number of epochs with no improvement after which
                        training will be stopped
  --tensor_height TENSOR_HEIGHT
                        Height of the tensor representing the image
  --tensor_width TENSOR_WIDTH
                        Width of the tensor representing the image
  --monitored_value MONITORED_VALUE
                        Metric name to be monitored
  --force_dataset_generation FORCE_DATASET_GENERATION
                        Boolean to force the dataset structure generation
  --fit_dataset_in_memory FIT_DATASET_IN_MEMORY
                        Boolean to load the entire dataset into memory instead
                        of opening new files with each request - results in
                        the reduction of I/O operations and time, but could
                        result in huge memory needs in case of a big dataset
  --augment_training_dataset AUGMENT_TRAINING_DATASET
                        Boolean to augment the training dataset with
                        rotations, shear and flips
  --tversky_alpha TVERSKY_ALPHA
                        ONLY FOR LOSS_FUNCTION == TVERSKY: Coefficient alpha
  --tversky_beta TVERSKY_BETA
                        ONLY FOR LOSS_FUNCTION == TVERSKY: Coefficient beta
  --dropout_rate_input DROPOUT_RATE_INPUT
                        Fraction of the input units of the input layer to drop
  --dropout_rate_hidden DROPOUT_RATE_HIDDEN
                        Fraction of the input units of the hidden layers to
                        drop
  --validation_set_percentage VALIDATION_SET_PERCENTAGE
                        If generating the dataset - Percentage of the entire
                        dataset to be used for the validation or detection in
                        the form of a decimal number
  --filter_by_classes FILTER_BY_CLASSES
                        If generating the dataset - Classes of interest. If
                        specified, only samples containing at least one of
                        them will be created. If filtering by multiple
                        classes, specify their values comma-separated (e.g.
                        "1,2,6" to filter by classes 1, 2 and 6)
  --backbone {ResNet50,ResNet101,ResNet152}
                        Backbone architecture
```

## Detection

Once you have a trained model, you can run the detection using the script
`detect.py`.

```
usage: detect.py [-h] --data_dir DATA_DIR [--model {U-Net,SegNet,DeepLab}]
                 [--weights_path WEIGHTS_PATH]
                 [--visualization_path VISUALIZATION_PATH]
                 [--batch_size BATCH_SIZE] [--seed SEED]
                 [--tensor_height TENSOR_HEIGHT] [--tensor_width TENSOR_WIDTH]
                 [--force_dataset_generation FORCE_DATASET_GENERATION]
                 [--fit_dataset_in_memory FIT_DATASET_IN_MEMORY]
                 [--validation_set_percentage VALIDATION_SET_PERCENTAGE]
                 [--filter_by_classes FILTER_BY_CLASSES]
                 [--backbone {ResNet50,ResNet101,ResNet152}]

Run detection

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Path to the directory containing images and labels
  --model {U-Net,SegNet,DeepLab}
                        Model architecture
  --weights_path WEIGHTS_PATH
                        Input weights path
  --visualization_path VISUALIZATION_PATH
                        Path to a directory where the detection visualizations
                        will be saved
  --batch_size BATCH_SIZE
                        The number of samples that will be propagated through
                        the network at once
  --seed SEED           Generator random seed
  --tensor_height TENSOR_HEIGHT
                        Height of the tensor representing the image
  --tensor_width TENSOR_WIDTH
                        Width of the tensor representing the image
  --force_dataset_generation FORCE_DATASET_GENERATION
                        Boolean to force the dataset structure generation
  --fit_dataset_in_memory FIT_DATASET_IN_MEMORY
                        Boolean to load the entire dataset into memory instead
                        of opening new files with each request - results in
                        the reduction of I/O operations and time, but could
                        result in huge memory needs in case of a big dataset
  --validation_set_percentage VALIDATION_SET_PERCENTAGE
                        If generating the dataset - Percentage of the entire
                        dataset to be used for the detection in the form of
                        a decimal number
  --filter_by_classes FILTER_BY_CLASSES
                        If generating the dataset - Classes of interest. If
                        specified, only samples containing at least one of
                        them will be created. If filtering by multiple
                        classes, specify their values comma-separated (e.g.
                        "1,2,6" to filter by classes 1, 2 and 6)
  --backbone {ResNet50,ResNet101,ResNet152}
                        Backbone architecture
```
