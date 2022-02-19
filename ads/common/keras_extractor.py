#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.model_info_extractor import ModelInfoExtractor
from ads.common.model_metadata import Framework


class KerasExtractor(ModelInfoExtractor):
    """Class that extract model metadata from keras models.

    Attributes
    ----------
    model: object
        The model to extract metadata from.
    estimator: object
        The estimator to extract metadata from.

    Methods
    -------
    framework(self) -> str
        Returns the framework of the model.
    algorithm(self) -> object
        Returns the algorithm of the model.
    version(self) -> str
        Returns the version of framework of the model.
    hyperparameter(self) -> dict
        Returns the hyperparameter of the model.
    """

    def __init__(self, model, estimator):
        self.model = model
        self.estimator = estimator

    def framework(self):
        """Extracts the framework of the model.

        Returns
        ----------
        str:
           The framework of the model.
        """
        return Framework.KERAS

    def algorithm(self):
        """Extracts the algorithm of the model.

        Returns
        ----------
        object:
           The algorithm of the model.
        """
        return self.estimator

    def version(self):
        """Extracts the framework version of the model.

        Returns
        ----------
        str:
           The framework version of the model.
        """
        from tensorflow import keras

        return keras.__version__

    def hyperparameter(self):
        """Extracts the hyperparameters of the model.

        Returns
        ----------
        dict:
           The hyperparameters of the model.
        """
        return self.model.get_config()
