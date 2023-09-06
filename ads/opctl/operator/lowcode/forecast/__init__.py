#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os

__version__ = "v1"

__type__ = "forecast"

# __conda__ = "forecast_p38_cpu_v1" # switch to slug when service conda env will be available

__conda__ = "cpu/forecast/1/forecast_v1"

__gpu__ = "no"  # yes/no

__keywords__ = ["Prophet", "AutoML", "ARIMA", "RNN", "LSTM"]

__operator_path__ = os.path.dirname(__file__)

__short_description__ = """
Forecasting operator, that leverages historical time series data to generate accurate
forecasts for future trends. Use `ads opctl operator info forecast`
to get more details about the forecasting operator.
"""
