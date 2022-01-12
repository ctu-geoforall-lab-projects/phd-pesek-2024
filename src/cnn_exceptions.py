#!/usr/bin/python3


class ModelConfigError(Exception):
    """Error to be raised in the case of wrong parameters of a model config."""

    pass


class DatasetError(Exception):
    """Error to be raised in the case of problems with the dataset."""

    pass


class LayerDefinitionError(Exception):
    """Error to be raised in the case of problems with the layer definition."""

    pass
