#!/usr/bin/python3

class ModelConfigError(Exception):
    """Model to be raised in the case of wrong parameters to build up a model.
    """

    pass


class DatasetError(Exception):
    """Model to be raised in the case of problems with the dataset.
    """

    pass


class LayerDefinitionError(Exception):
    """Model to be raised in the case of problems with the layer definition.
    """

    pass
