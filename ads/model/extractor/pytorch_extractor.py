#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.model.extractor.model_info_extractor import ModelInfoExtractor
from ads.model.model_metadata import Framework
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


class PyTorchExtractor(ModelInfoExtractor):
    """Class that extract model metadata from pytorch models.

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

    def __init__(self, model):
        self.model = model

    @property
    def framework(self):
        """Extracts the framework of the model.

        Returns
        ----------
        str:
           The framework of the model.
        """
        return Framework.PYTORCH

    @property
    def algorithm(self):
        """Extracts the algorithm of the model.

        Returns
        ----------
        object:
           The algorithm of the model.
        """
        return self.model.__class__.__name__

    @property
    @runtime_dependency(module="torch", install_from=OptionalDependency.PYTORCH)
    def version(self):
        """Extracts the framework version of the model.

        Returns
        ----------
        str:
           The framework version of the model.
        """
        return torch.__version__

    @property
    def hyperparameter(self):
        """Extracts the hyperparameters of the model.

        Returns
        ----------
        dict:
           The hyperparameters of the model.
        """
        return None


class PytorchExtractor(PyTorchExtractor):
    pass
