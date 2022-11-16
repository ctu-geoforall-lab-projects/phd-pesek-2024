import os
import sys
import pytest
import filecmp

from difflib import unified_diff

from train import main as train


def report_file(identifier):
    """Report the inconsistency of two files.

    To be called when output comparison assert fails.

    :param identifier: identifier of the training setting
    :return: string message reporting the content of the new output
    """
    sys.stdout.write(f'Output for setting {identifier} not consistent with ' \
                      'the stored results. The diff is as follows:\n\n')

    with open(f'/tmp/{identifier}.txt') as left:
        with open(f'src/test/consistency_outputs/{identifier}.txt') as right:
            sys.stdout.writelines(unified_diff(left.readlines(), right.readlines()))

    return f'Inconsistency in outputs of setting {identifier}'


class TestCmd:
    def test_001_main_architectures(self, capsys):
        """Test the consistency of a small cloud classification sample.

        Test all architectures with and without droput.

        :param capsys: a builtin pytest fixture that ispassed into any test to
                       capture stdin/stdout
        """
        training_data_dir = os.path.join('/tmp', 'training_data',
                                         'training_set_clouds_multiclass')
        # TODO: Add continue

        # tests for architectures without backbone models
        for architecture in ('U-Net', 'SegNet'):
            for dropout in (0, 0.5):
                identifier = f'{architecture.lower()}_drop{dropout}_categorical_crossentropy'
                train(operation='train',
                      model=architecture,
                      data_dir=training_data_dir,
                      output_dir=f'/tmp/output_{identifier}',
                      model_fn=f'/tmp/output_{identifier}/model.h5',
                      visualization_path=f'/tmp/output_{identifier}',
                      nr_epochs=2,
                      dropout_rate_hidden=dropout,
                      val_set_pct=0.5,
                      monitored_value='val_loss',
                      loss_function='categorical_crossentropy',
                      tensor_shape=(256, 256),
                      filter_by_class='1,2',
                      seed=1,
                      name=identifier,
                      verbose=0)

                cap = capsys.readouterr()

                with open(f'/tmp/{identifier}.txt', 'w') as out:
                    out.write(cap.out)

                assert filecmp.cmp(f'/tmp/{identifier}.txt', f'src/test/consistency_outputs/{identifier}.txt'), report_file(identifier)

        # tests for architectures with backbone models
        architecture = 'DeepLab'
        for backbone in ('ResNet50', 'ResNet101', 'ResNet152'):
            for dropout in (0, 0.5):
                identifier = f'{architecture.lower()}_drop{dropout}_{backbone}_categorical_crossentropy'
                train(operation='train',
                      model=architecture,
                      data_dir=training_data_dir,
                      output_dir=f'/tmp/output_{identifier}',
                      model_fn=f'/tmp/output_{identifier}/model.h5',
                      visualization_path=f'/tmp/output_{identifier}',
                      nr_epochs=2,
                      dropout_rate_hidden=dropout,
                      val_set_pct=0.5,
                      monitored_value='val_loss',
                      loss_function='categorical_crossentropy',
                      tensor_shape=(256, 256),
                      filter_by_class='1,2',
                      seed=1,
                      backbone=backbone,
                      name=identifier,
                      verbose=0)

                cap = capsys.readouterr()

                with open(f'/tmp/{identifier}.txt', 'w') as out:
                    out.write(cap.out)

                assert filecmp.cmp(f'/tmp/{identifier}.txt', f'src/test/consistency_outputs/{identifier}.txt'), report_file(identifier)

    def test_002_loss(self, capsys):
        """Test the consistency of a small cloud classification sample.

        Test the consistency of loss functions.

        :param capsys: a builtin pytest fixture that ispassed into any test to
                       capture stdin/stdout
        """
        training_data_dir = os.path.join('/tmp', 'training_data',
                                         'training_set_clouds_multiclass')
        # TODO: Add binary loss

        for loss in ('categorical_crossentropy', 'dice'):
            identifier = f'u-net_drop0_{loss}'
            train(operation='train',
                  model='U-Net',
                  data_dir=training_data_dir,
                  output_dir=f'/tmp/output_{identifier}',
                  model_fn=f'/tmp/output_{identifier}/model.h5',
                  visualization_path=f'/tmp/output_{identifier}',
                  nr_epochs=2,
                  dropout_rate_hidden=0,
                  val_set_pct=0.5,
                  monitored_value='val_loss',
                  loss_function=loss,
                  tensor_shape=(256, 256),
                  filter_by_class='1,2',
                  seed=1,
                  name=identifier,
                  verbose=0)

            cap = capsys.readouterr()

            with open(f'/tmp/{identifier}.txt', 'w') as out:
                out.write(cap.out)

            assert filecmp.cmp(f'/tmp/{identifier}.txt', f'src/test/consistency_outputs/{identifier}.txt'), report_file(identifier)

        # test tversky
        for alpha, beta in ((0.3, 0.7), (0.7, 0.3)):
            identifier = f'u-net_drop0_tversky_{alpha}_{beta}'
            train(operation='train',
                  model='U-Net',
                  data_dir=training_data_dir,
                  output_dir=f'/tmp/output_{identifier}',
                  model_fn=f'/tmp/output_{identifier}/model.h5',
                  visualization_path=f'/tmp/output_{identifier}',
                  nr_epochs=2,
                  dropout_rate_hidden=0,
                  val_set_pct=0.5,
                  monitored_value='val_loss',
                  loss_function='tversky',
                  tensor_shape=(256, 256),
                  filter_by_class='1,2',
                  seed=1,
                  tversky_alpha=alpha,
                  tversky_beta=beta,
                  name=identifier,
                  verbose=0)

            cap = capsys.readouterr()

            with open(f'/tmp/{identifier}.txt', 'w') as out:
                out.write(cap.out)

            assert filecmp.cmp(f'/tmp/{identifier}.txt', f'src/test/consistency_outputs/{identifier}.txt'), report_file(identifier)

    def test_003_augmentation(self, capsys):
        """Test the consistency of a small cloud classification sample.

        Test all architectures with and without droput.

        :param capsys: a builtin pytest fixture that ispassed into any test to
                       capture stdin/stdout
        """
        training_data_dir = os.path.join('/tmp', 'training_data',
                                         'training_set_clouds_multiclass')

        identifier = 'u-net_drop0_categorical_crossentropy_augment'
        train(operation='train',
              model='U-Net',
              data_dir=training_data_dir,
              output_dir=f'/tmp/output_{identifier}',
              model_fn=f'/tmp/output_{identifier}/model.h5',
              visualization_path=f'/tmp/output_{identifier}',
              nr_epochs=2,
              dropout_rate_hidden=0,
              val_set_pct=0.5,
              monitored_value='val_loss',
              loss_function='categorical_crossentropy',
              tensor_shape=(256, 256),
              filter_by_class='1,2',
              seed=1,
              augment=True,
              force_dataset_generation=True,
              name=identifier,
              verbose=0)

        cap = capsys.readouterr()

        with open(f'/tmp/{identifier}.txt', 'w') as out:
            out.write(cap.out)

        assert filecmp.cmp(f'/tmp/{identifier}.txt', f'src/test/consistency_outputs/{identifier}.txt'), report_file(identifier)
