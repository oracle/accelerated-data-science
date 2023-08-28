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


MAX_COLUMNS_AUTOMLX = 15