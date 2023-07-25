#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

__version__ = "v1"

__name__ = "forecast"

# __conda__ = "forecast_p38_cpu_v1" # switch to slug when service conda env will be available

__conda__ = "cpu/Forecasting Operator/1.0/forecast_p38_cpu_v1"

__keywords__ = ["Prophet", "AutoML", "ARIMA", "RNN", "LSTM"]

__short_description__ = """
Forecasting operator, that leverages historical time series data to generate accurate
forecasts for future trends. Use `ads opctl operator info forecast`
to get more details about the forecasting operator.
"""

__description__ = """
Forecasting operator, that leverages historical time series data to generate accurate
forecasts for future trends. This operator aims to simplify and expedite the data science process by
automating the selection of appropriate models and hyperparameters, as well as identifying relevant
features for a given prediction task.

The detained information, explaining how to deal with the forecasting operator will be placed here.
"""
