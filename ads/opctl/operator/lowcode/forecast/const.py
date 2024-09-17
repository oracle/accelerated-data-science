#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.extended_enum import ExtendedEnumMeta
from ads.opctl.operator.lowcode.common.const import DataColumns


class SupportedModels(str, metaclass=ExtendedEnumMeta):
    """Supported forecast models."""

    Prophet = "prophet"
    Arima = "arima"
    NeuralProphet = "neuralprophet"
    MLForecast = "mlforecast"
    AutoMLX = "automlx"
    AutoTS = "autots"
    Auto = "auto"


class SpeedAccuracyMode(str, metaclass=ExtendedEnumMeta):
    """
    Enum representing different modes based on time taken and accuracy for explainability.
    """

    HIGH_ACCURACY = "HIGH_ACCURACY"
    BALANCED = "BALANCED"
    FAST_APPROXIMATE = "FAST_APPROXIMATE"
    ratio = dict()
    ratio[HIGH_ACCURACY] = 1  # 100 % data used for generating explanations
    ratio[BALANCED] = 0.5  # 50 % data used for generating explanations
    ratio[FAST_APPROXIMATE] = 0  # constant


class SupportedMetrics(str, metaclass=ExtendedEnumMeta):
    """Supported forecast metrics."""

    MAPE = "MAPE"
    RMSE = "RMSE"
    MSE = "MSE"
    SMAPE = "sMAPE"
    WMAPE = "wMAPE"
    R2 = "r2"
    EXPLAINED_VARIANCE = "Explained Variance"
    MEAN_MAPE = "Mean MAPE"
    MEAN_RMSE = "Mean RMSE"
    MEAN_MSE = "Mean MSE"
    MEAN_SMAPE = "Mean sMAPE"
    MEAN_WMAPE = "Mean wMAPE"
    MEAN_R2 = "Mean r2"
    MEAN_EXPLAINED_VARIANCE = "Mean Explained Variance"
    MEDIAN_MAPE = "Median MAPE"
    MEDIAN_RMSE = "Median RMSE"
    MEDIAN_MSE = "Median MSE"
    MEDIAN_SMAPE = "Median sMAPE"
    MEDIAN_WMAPE = "Median wMAPE"
    MEDIAN_R2 = "Median r2"
    MEDIAN_EXPLAINED_VARIANCE = "Median Explained Variance"
    ELAPSED_TIME = "Elapsed Time"


class ForecastOutputColumns(str, metaclass=ExtendedEnumMeta):
    """The column names for the forecast.csv output file"""

    DATE = "Date"
    SERIES = DataColumns.Series
    INPUT_VALUE = "input_value"
    FITTED_VALUE = "fitted_value"
    FORECAST_VALUE = "forecast_value"
    UPPER_BOUND = "upper_bound"
    LOWER_BOUND = "lower_bound"


AUTOMLX_METRIC_MAP = {
    "smape": "neg_sym_mean_abs_percent_error",
    "mape": "neg_sym_mean_abs_percent_error",
    "mase": "neg_mean_abs_scaled_error",
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
    "rmse": "neg_root_mean_squared_error",
}

MAX_COLUMNS_AUTOMLX = 15
DEFAULT_TRIALS = 10
SUMMARY_METRICS_HORIZON_LIMIT = 10
PROPHET_INTERNAL_DATE_COL = "ds"
RENDER_LIMIT = 5000
AUTO_SELECT = "auto-select"
