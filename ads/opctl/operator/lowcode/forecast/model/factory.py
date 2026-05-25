#!/usr/bin/env python

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl.operator.lowcode.common.const import DataColumns

from ..const import (
    AUTO_SELECT,
    AUTO_SELECT_SERIES,
    AUTO_SELECT_SERIES_SELECTION_STRATEGY_KEY,
    AutoSelectSeriesSelectionStrategy,
    DEFAULT_AUTO_SELECT_SERIES_BACKTESTING_MODELS,
    ForecastOutputColumns,
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
from .meta_features import build_meta_features
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
            auto_select_series_strategy = cls.get_auto_select_series_selection_strategy(
                operator_config
            )
            operator_config.spec.series_selection_strategy = (
                auto_select_series_strategy
            )
            if (
                auto_select_series_strategy
                == AutoSelectSeriesSelectionStrategy.BACKTESTING
            ):
                model_type = cls.auto_select_series_backtesting_model(
                    datasets, operator_config
                )
            else:
                model_type = cls.auto_select_series_meta_learning_model(
                    datasets, operator_config
                )
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

    @classmethod
    def auto_select_series_meta_learning_model(
            cls, datasets: ForecastDatasets, operator_config: ForecastOperatorConfig
    ) -> str:
        """
        Selects the best model for each series independently using meta-learning and returns the
        most common winning model so the factory can instantiate a concrete operator model type.
        """
        selector = MetaSelector()
        spec = datasets.historical_data.spec

        additional_df = datasets.additional_data.data.reset_index()
        historical_df = datasets.historical_data.data.reset_index()
        series_col = ForecastOutputColumns.SERIES
        timestamp_col = spec.datetime_column.name
        dataset = historical_df.merge(
            additional_df, on=[series_col, timestamp_col], how="left"
        )
        meta_feature_table = build_meta_features(
            dataset,
            target_col=spec.target_column,
            series_col=series_col,
            timestamp_col=timestamp_col,
            horizon=spec.horizon,
            frequency_hint=datasets.historical_data.freq,
        )

        meta_features = selector.select_best_model(meta_feature_table)
        operator_config.spec.meta_features = meta_features
        return meta_features["selected_model"].mode().iloc[0]

    @classmethod
    def auto_select_series_backtesting_model(
            cls, datasets: ForecastDatasets, operator_config: ForecastOperatorConfig
    ) -> str:
        """
        Selects the best model for each series independently through backtesting and returns the
        most common winning model so the factory can instantiate a concrete operator model type.
        """
        all_models = operator_config.spec.model_kwargs.get(
            "model_list", DEFAULT_AUTO_SELECT_SERIES_BACKTESTING_MODELS
        )
        if AUTO_SELECT_SERIES in all_models:
            all_models = [
                model
                for model in all_models
                if model != AUTO_SELECT_SERIES
            ]
        num_backtests = operator_config.spec.model_kwargs.get("num_backtests", 5)
        model_evaluator = ModelEvaluator(all_models, num_backtests)
        series_model_selection = model_evaluator.find_best_model_per_series(
            datasets, operator_config
        )
        operator_config.spec.series_model_selection = series_model_selection
        return series_model_selection["selected_model"].mode().iloc[0]

    @staticmethod
    def get_auto_select_series_selection_strategy(
            operator_config: ForecastOperatorConfig,
    ) -> str:
        """Returns the configured strategy for auto-select-series."""
        strategy = operator_config.spec.model_kwargs.get(
            AUTO_SELECT_SERIES_SELECTION_STRATEGY_KEY
        )
        if strategy == AutoSelectSeriesSelectionStrategy.BACKTESTING:
            return AutoSelectSeriesSelectionStrategy.BACKTESTING
        return AutoSelectSeriesSelectionStrategy.META_LEARNING
