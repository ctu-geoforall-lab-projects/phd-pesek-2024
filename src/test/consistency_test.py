import pytest
import filecmp

from train import main as train


def print_file(file):
    with open(file) as open_file:
        return open_file.read()


class TestCmd:
    def test_001_clouds(self, capsys):
        """Test the consistency of a small cloud classification sample."""
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

        # with open(f'/tmp/{identifier}.txt', 'w') as out:
        with open(f'/tmp/out.txt', 'w') as out:
            out.write(cap.out)

#                 assert filecmp.cmp(f'/tmp/{identifier}.txt', f'src/test/consistency_outputs/{identifier}.txt'), print_file(f'/tmp/{identifier}.txt')

