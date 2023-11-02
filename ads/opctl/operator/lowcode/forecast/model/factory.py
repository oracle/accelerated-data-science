#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ..const import SupportedModels
from ..operator_config import ForecastOperatorConfig
from .arima import ArimaOperatorModel
from .automlx import AutoMLXOperatorModel
from .autots import AutoTSOperatorModel
from .base_model import ForecastOperatorBaseModel
from .neuralprophet import NeuralProphetOperatorModel
from .prophet import ProphetOperatorModel
from ..utils import select_auto_model
from .forecast_datasets import ForecastDatasets

class UnSupportedModelError(Exception):
    def __init__(self, model_type: str):
        super().__init__(
            f"Model: `{model_type}` "
            f"is not supported. Supported models: {SupportedModels.values}"
        )


class ForecastOperatorModelFactory:
    """
    The factory class helps to instantiate proper model operator based on the model type.
    """

    _MAP = {
        SupportedModels.Prophet: ProphetOperatorModel,
        SupportedModels.Arima: ArimaOperatorModel,
        SupportedModels.NeuralProphet: NeuralProphetOperatorModel,
        SupportedModels.AutoMLX: AutoMLXOperatorModel,
        SupportedModels.AutoTS: AutoTSOperatorModel
    }

    @classmethod
    def get_model(
        cls, operator_config: ForecastOperatorConfig, datasets: ForecastDatasets
    ) -> ForecastOperatorBaseModel:
        """
        Gets the forecasting operator model based on the model type.

        Parameters
        ----------
        operator_config: ForecastOperatorConfig
            The forecasting operator config.
        datasets: ForecastDatasets
            Datasets for predictions

        Returns
        -------
        ForecastOperatorBaseModel
            The forecast operator model.

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
