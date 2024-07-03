#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ..constant import SupportedModels
from ..operator_config import RecommenderOperatorConfig
from .base_model import RecommenderOperatorBaseModel
from .recommender_dataset import RecommenderDatasets
from .svd import SVDOperatorModel

class UnSupportedModelError(Exception):
    def __init__(self, model_type: str):
        super().__init__(
            f"Model: `{model_type}` "
            f"is not supported. Supported models: {SupportedModels.values}"
        )


class RecommenderOperatorModelFactory:
    """
    The factory class helps to instantiate proper model operator based on the model type.
    """

    _MAP = {
        SupportedModels.SVD: SVDOperatorModel
    }

    @classmethod
    def get_model(
        cls, operator_config: RecommenderOperatorConfig, datasets: RecommenderDatasets
    ) -> RecommenderOperatorBaseModel:
        """
        Gets the operator model based on the model type.

        Parameters
        ----------
        operator_config: RecommenderOperatorConfig
            The recommender detection operator config.

        datasets: RecommenderDatasets
            Datasets for finding recommender

        Returns
        -------
        RecommenderOperatorBaseModel
            The recommender detection operator model.

        Raises
        ------
        UnSupportedModelError
            In case of not supported model.
        """
        model_type = SupportedModels.SVD
        if model_type not in cls._MAP:
            raise UnSupportedModelError(model_type)
        return cls._MAP[model_type](config=operator_config, datasets=datasets)
