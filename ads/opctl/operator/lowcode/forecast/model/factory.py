#!/usr/bin/env python

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl.operator.lowcode.common.transformations import Transformations

from ..const import (
    AUTO_SELECT,
    AUTO_SELECT_SERIES,
    TROUBLESHOOTING_GUIDE,
    SpeedAccuracyMode,
    SupportedModels,
)
from ..meta_selector import MetaSelector
from ..model_evaluator import ModelEvaluator
from ..operator_config import ForecastOperatorConfig
from .arima import ArimaOperatorModel
from .automlx import AutoMLXOperatorModel
from .autots import AutoTSOperatorModel
from .base_model import ForecastOperatorBaseModel
from .forecast_datasets import ForecastDatasets
from .lgbforecast import LGBForecastOperatorModel
from .neuralprophet import NeuralProphetOperatorModel
from .prophet import ProphetOperatorModel
from .xgbforecast import XGBForecastOperatorModel
from .theta import ThetaOperatorModel
from .ets import ETSOperatorModel


class UnSupportedModelError(Exception):
    def __init__(self, model_type: str):
        super().__init__(
            f"Model: `{model_type}` "
            f"is not supported. Supported models: {SupportedModels.values()}"
            f"\nPlease refer to the troubleshooting guide at {TROUBLESHOOTING_GUIDE} for resolution steps."
        )


class ForecastOperatorModelFactory:
    """
    The factory class helps to instantiate proper model operator based on the model type.
    """

    _MAP = {
        SupportedModels.Prophet: ProphetOperatorModel,
        SupportedModels.Arima: ArimaOperatorModel,
        SupportedModels.NeuralProphet: NeuralProphetOperatorModel,
        SupportedModels.LGBForecast: LGBForecastOperatorModel,
        SupportedModels.XGBForecast: XGBForecastOperatorModel,
        SupportedModels.AutoMLX: AutoMLXOperatorModel,
        SupportedModels.AutoTS: AutoTSOperatorModel,
        SupportedModels.Theta: ThetaOperatorModel,
        SupportedModels.ETSForecaster: ETSOperatorModel,
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

        if model_type == AUTO_SELECT_SERIES:
            # Initialize MetaSelector for series-specific model selection
            selector = MetaSelector()
            # Create a Transformations instance
            transformer = Transformations(dataset_info=datasets.historical_data.spec)

            # Calculate meta-features
            meta_features = selector.select_best_model(
                meta_features_df=transformer.build_fforms_meta_features(
                    data=datasets.historical_data.raw_data,
                    target_col=datasets.historical_data.spec.target_column,
                    group_cols=datasets.historical_data.spec.target_category_columns
                )
            )
            # Get the most common model as default
            model_type = meta_features['selected_model'].mode().iloc[0]
            # Store the series-specific model selections in the config for later use
            operator_config.spec.meta_features = meta_features
            operator_config.spec.model_kwargs = {}

        elif model_type == AUTO_SELECT:
            model_type = cls.auto_select_model(datasets, operator_config)
            operator_config.spec.model_kwargs = {}
            # set the explanations accuracy mode to AUTOMLX if the selected model is automlx
            if (
                model_type == SupportedModels.AutoMLX
                and operator_config.spec.explanations_accuracy_mode
                == SpeedAccuracyMode.FAST_APPROXIMATE
            ):
                operator_config.spec.explanations_accuracy_mode = SpeedAccuracyMode.AUTOMLX
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
        all_models = operator_config.spec.model_kwargs.get(
            "model_list", cls._MAP.keys()
        )
        num_backtests = operator_config.spec.model_kwargs.get("num_backtests", 5)
        sample_ratio = operator_config.spec.model_kwargs.get("sample_ratio", 0.20)
        model_evaluator = ModelEvaluator(all_models, num_backtests, sample_ratio)
        return model_evaluator.find_best_model(datasets, operator_config)
