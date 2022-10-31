import os
import sys
import configparser
import filecmp
import pytest

from shutil import rmtree
from pathlib import Path

from train import main as train


def are_dir_trees_equal(dir1, dir2):
    """
    Taken from https://stackoverflow.com/questions/4187564/recursively-compare-two-directories-to-ensure-they-have-the-same-files-and-subdi
    Compare two directories recursively. Files in each directory are
    assumed to be equal if their names and contents are equal.

    @param dir1: First directory path
    @param dir2: Second directory path

    @return: True if the directory trees are the same and
        there were no errors while accessing the directories or files,
        False otherwise.
    """
    dirs_cmp = filecmp.dircmp(dir1, dir2)
    if len(dirs_cmp.left_only)>0 or len(dirs_cmp.right_only)>0 or \
        len(dirs_cmp.funny_files)>0:
        return False
    (_, mismatch, errors) = filecmp.cmpfiles(
        dir1, dir2, dirs_cmp.common_files, shallow=False)
    if len(mismatch)>0 or len(errors)>0:
        return False
    for common_dir in dirs_cmp.common_dirs:
        new_dir1 = Path(dir1) / common_dir
        new_dir2 = Path(dir2 / common_dir)
        if not are_dir_trees_equal(new_dir1, new_dir2):
            return False
    return True


class TestCmd:
    def test_001_clouds(self, capsys):
        """Test the consistency of a small cloud classification sample."""
        # TODO: Add augment, continue, val_losses
        # for architecture in ('U-Net', 'SegNet', 'DeepLab',):
        for architecture in ('DeepLab',):
            for dropout in (0, 0.5):
                train(operation='train',
                      model=architecture,
                      data_dir='/tmp/training_data/training_set_clouds_multiclass',
                      output_dir=f'/tmp/output_{architecture.lower()}_{dropout}',
                      model_fn=f'/tmp/output_{architecture.lower()}_{dropout}/model.h5',
                      visualization_path=f'/tmp/output_{architecture.lower()}_{dropout}/visualizations/',
                      nr_epochs=3,
                      dropout_rate_hidden=dropout,
                      val_set_pct=0.5,
                      monitored_value='val_loss',
                      loss_function='categorical_crossentropy',
                      # tensor_shape=(32, 32),
                      tensor_shape=(256, 256),
                      filter_by_class='1,2',
                      seed=1)

        cap = capsys.readouterr()
        
        with open('/tmp/out.out', 'w') as out:
            out.write(cap.out)

