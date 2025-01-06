#!/usr/bin/env python

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.decorator.runtime_dependency import (
    OptionalDependency,
    runtime_dependency,
)
from ads.model.extractor.model_info_extractor import ModelInfoExtractor
from ads.model.model_metadata import Framework


class EmbeddingONNXExtractor(ModelInfoExtractor):
    """Class that extract model metadata from EmbeddingONNXModel models.

    Attributes
    ----------
    model: object
        The model to extract metadata from.

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

    def __init__(self, model=None):
        self.model = model

    @property
    def framework(self):
        """Extracts the framework of the model.

        Returns
        ----------
        str:
           The framework of the model.
        """
        return Framework.EMBEDDING_ONNX

    @property
    def algorithm(self):
        """Extracts the algorithm of the model.

        Returns
        ----------
        object:
           The algorithm of the model.
        """
        return "Embedding_ONNX"

    @property
    @runtime_dependency(module="onnxruntime", install_from=OptionalDependency.ONNX)
    def version(self):
        """Extracts the framework version of the model.

        Returns
        ----------
        str:
           The framework version of the model.
        """
        return onnxruntime.__version__

    @property
    def hyperparameter(self):
        """Extracts the hyperparameters of the model.

        Returns
        ----------
        dict:
           The hyperparameters of the model.
        """
        return None
