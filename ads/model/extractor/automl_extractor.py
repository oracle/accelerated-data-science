#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.decorator.deprecate import deprecated
from ads.model.extractor.model_info_extractor import ModelInfoExtractor
from ads.model.model_metadata import Framework


class AutoMLExtractor(ModelInfoExtractor):
    """Class that extract model metadata from automl models.

    Attributes
    ----------
    model: object
        The model to extract metadata from.
    estimator: object
        The estimator to extract metadata from.
    """

    @deprecated(
        details="Working with AutoML has moved from within ADS to working directly with the AutoMLx library. AutoMLx are preinstalled in conda pack automlx_p38_cpu_v2 and later, and can now be updated independently of ADS. AutoMLx documentation may be found at https://docs.oracle.com/en-us/iaas/tools/automlx/latest/html/multiversion/v23.1.1/index.html. Notebook examples are in Oracle's samples repository: https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/notebook_examples and a migration tutorial can be found at https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_training/automl/quick_start.html .",
        raise_error=True,
    )
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
        return Framework.ORACLE_AUTOML

    @property
    def algorithm(self):
        """Extracts the algorithm of the model.

        Returns
        ----------
        object:
           The algorithm of the model.
        """
        return "ensemble"

    @property
    def version(self):
        """Extracts the framework version of the model.

        Returns
        ----------
        str:
           The framework version of the model.
        """
        import automl

        return automl.__version__

    @property
    def hyperparameter(self):
        """Extracts the hyperparameters of the model.

        Returns
        ----------
        dict:
           The hyperparameters of the model.
        """
        return None
