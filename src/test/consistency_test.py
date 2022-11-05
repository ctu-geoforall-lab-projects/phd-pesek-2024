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

    return 'Inconsistency in outputs of setting {identifier}'


class TestCmd:
    def test_001_clouds(self, capsys):
        """Test the consistency of a small cloud classification sample.

        :param capsys: a builtin pytest fixture that ispassed into any test to
                       capture stdin/stdout
        """
        # TODO: Add augment, continue, val_losses
        training_data_dir = '/tmp/training_data/training_set_clouds_multiclass'

        for architecture in ('U-Net', 'SegNet', 'DeepLab',):
            for dropout in (0, 0.5):
                identifier = f'{architecture.lower()}_drop{dropout}'
                train(operation='train',
                      model=architecture,
                      data_dir=training_data_dir,
                      output_dir=f'/tmp/output_{identifier}',
                      model_fn=f'/tmp/output_{identifier}/model.h5',
                      visualization_path=f'/tmp/output_{identifier}',
                      nr_epochs=3,
                      dropout_rate_hidden=dropout,
                      val_set_pct=0.5,
                      monitored_value='val_loss',
                      loss_function='categorical_crossentropy',
                      tensor_shape=(256, 256),
                      filter_by_class='1,2',
                      seed=1,
                      verbose=0)

                cap = capsys.readouterr()

                with open(f'/tmp/{identifier}.txt', 'w') as out:
                    out.write(cap.out)

                assert filecmp.cmp(f'/tmp/{identifier}.txt', f'src/test/consistency_outputs/{identifier}.txt'), report_file(identifier)

