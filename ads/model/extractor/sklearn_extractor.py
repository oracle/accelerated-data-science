#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import logging
import re
from collections import defaultdict

from ads.model.extractor.model_info_extractor import (
    ModelInfoExtractor,
    normalize_hyperparameter,
)
from ads.model.model_metadata import Framework


class SklearnExtractor(ModelInfoExtractor):
    """Class that extract model metadata from sklearn models.

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
        return Framework.SCIKIT_LEARN

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
    def version(self):
        """Extracts the framework version of the model.

        Returns
        ----------
        str:
           The framework version of the model.
        """
        import sklearn

        return sklearn.__version__

    @property
    def hyperparameter(self):
        """Extracts the hyperparameters of the model.

        Returns
        ----------
        dict:
           The hyperparameters of the model.
        """
        if hasattr(self.model, "get_params"):
            hp_dict = self.model.get_params()
            # make shallow copy to avoid modifying the model object
            new_dict = hp_dict.copy()
            # handle sklearn pipeline case
            if "steps" in hp_dict:
                new_dict["steps"] = defaultdict(list)
                for i, (k, v) in enumerate(hp_dict["steps"]):
                    new_dict["steps"][i] = {k: re.sub("[()]", "", str(v))}
                    new_dict[k] = re.sub("[()]", "", str(v))
            # handle sklearn model selection case
            elif "param_grid" in hp_dict:
                new_dict["estimator"] = str(hp_dict["estimator"])
                new_dict["param_grid"] = defaultdict(list)
                for k, v in hp_dict["param_grid"].items():
                    new_dict["param_grid"][k] = v.tolist()
                new_dict.update(self.model.best_params_)

            return normalize_hyperparameter(new_dict)
        else:
            # for onnx model case.
            logging.warning(
                "Cannot extract the hyperparameters from this model automatically."
            )
            return {}
