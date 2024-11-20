#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.model.extractor.model_info_extractor import ModelInfoExtractor


class EmbeddingONNXExtractor(ModelInfoExtractor):
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
        pass

    @property
    def algorithm(self):
        """Extracts the algorithm of the model.

        Returns
        ----------
        object:
           The algorithm of the model.
        """
        pass

    @property
    def version(self):
        """Extracts the framework version of the model.

        Returns
        ----------
        str:
           The framework version of the model.
        """
        pass

    @property
    def hyperparameter(self):
        """Extracts the hyperparameters of the model.

        Returns
        ----------
        dict:
           The hyperparameters of the model.
        """
        pass
