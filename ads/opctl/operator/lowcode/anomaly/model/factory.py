#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ..const import SupportedModels
from ..operator_config import AnomalyOperatorConfig
from .automlx import AutoMLXOperatorModel
from .autots import AutoTSOperatorModel
from ads.opctl.operator.lowcode.anomaly.utils import select_auto_model

# from .tods import TODSOperatorModel
from .base_model import AnomalyOperatorBaseModel
from .anomaly_dataset import AnomalyDatasets


class UnSupportedModelError(Exception):
    def __init__(self, model_type: str):
        super().__init__(
            f"Model: `{model_type}` "
            f"is not supported. Supported models: {SupportedModels.values}"
        )


class AnomalyOperatorModelFactory:
    """
    The factory class helps to instantiate proper model operator based on the model type.
    """

    _MAP = {
        SupportedModels.AutoMLX: AutoMLXOperatorModel,
        # SupportedModels.TODS: TODSOperatorModel,
        SupportedModels.AutoTS: AutoTSOperatorModel,
    }

    @classmethod
    def get_model(
        cls, operator_config: AnomalyOperatorConfig, datasets: AnomalyDatasets
    ) -> AnomalyOperatorBaseModel:
        """
        Gets the operator model based on the model type.

        Parameters
        ----------
        operator_config: AnomalyOperatorConfig
            The anomaly detection operator config.

        datasets: AnomalyDatasets
            Datasets for finding anomaly

        Returns
        -------
        AnomalyOperatorBaseModel
            The anomaly detection operator model.

        Raises
        ------
        UnSupportedModelError
            In case of not supported model.
        """
        model_type = operator_config.spec.model
        if model_type == "auto":
            model_type = select_auto_model(datasets, operator_config)
        if model_type not in cls._MAP:
            raise UnSupportedModelError(model_type)
        return cls._MAP[model_type](config=operator_config, datasets=datasets)
