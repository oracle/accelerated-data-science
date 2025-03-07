#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ..const import SupportedModels, AUTO_SELECT
from ..operator_config import ForecastOperatorConfig
from .arima import ArimaOperatorModel
from .automlx import AutoMLXOperatorModel
from .autots import AutoTSOperatorModel
from .base_model import ForecastOperatorBaseModel
from .neuralprophet import NeuralProphetOperatorModel
from .prophet import ProphetOperatorModel
from .forecast_datasets import ForecastDatasets
from .ml_forecast import MLForecastOperatorModel
from ..model_evaluator import ModelEvaluator

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
        SupportedModels.MLForecast: MLForecastOperatorModel,
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
        if model_type == AUTO_SELECT:
            model_type = cls.auto_select_model(datasets, operator_config)
            operator_config.spec.model_kwargs = dict()
        if model_type not in cls._MAP:
            raise UnSupportedModelError(model_type)
        return cls._MAP[model_type](config=operator_config, datasets=datasets)

    @classmethod
    def auto_select_model(
            cls, datasets: ForecastDatasets, operator_config: ForecastOperatorConfig
    ) -> str:
        """
        Selects AutoMLX or Arima model based on column count.

        If the number of columns is less than or equal to the maximum allowed for AutoMLX,
        returns 'AutoMLX'. Otherwise, returns 'Arima'.

        Parameters
        ------------
        datasets:  ForecastDatasets
                Datasets for predictions

        Returns
        --------
        str
            The type of the model.
        """
        all_models = operator_config.spec.model_kwargs.get("model_list", cls._MAP.keys())
        num_backtests = operator_config.spec.model_kwargs.get("num_backtests", 5)
        sample_ratio = operator_config.spec.model_kwargs.get("sample_ratio", 0.20)
        model_evaluator = ModelEvaluator(all_models, num_backtests, sample_ratio)
        return model_evaluator.find_best_model(datasets, operator_config)