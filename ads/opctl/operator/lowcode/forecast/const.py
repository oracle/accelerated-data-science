#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.extended_enum import ExtendedEnumMeta


class SupportedModels(str, metaclass=ExtendedEnumMeta):
    """Supported forecast models."""

    Prophet = "prophet"
    Arima = "arima"
    NeuralProphet = "neuralprophet"
    AutoMLX = "automlx"


class SupportedMetrics(str, metaclass=ExtendedEnumMeta):
    """Supported forecast metrics."""

    MAPE = "mape"
    RMSE = "rmse"
    MSE = "mse"
    SMAPE = "smape"


automlx_metric_dict = {
    "smape": "neg_sym_mean_abs_percent_error",
    "mape": "neg_sym_mean_abs_percent_error",
    "mase": "neg_mean_abs_scaled_error",
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
    "rmse": "neg_root_mean_squared_error",
}
