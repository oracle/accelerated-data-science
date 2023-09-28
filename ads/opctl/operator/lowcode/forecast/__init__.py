#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os

__operator_path__ = os.path.dirname(__file__)

__type__ = os.path.basename(__operator_path__.rstrip("/"))

__version__ = "v1"

__conda__ = f"{__type__}_{__version__}"

__conda_type__ = "custom"  # service/custom

__gpu__ = "no"  # yes/no

__keywords__ = ["Prophet", "AutoML", "ARIMA", "RNN", "LSTM"]

__backends__ = ["job", "dataflow"]  # job/dataflow/


__short_description__ = """
Forecasting operator, that leverages historical time series data to generate accurate
forecasts for future trends. Use `ads operator info forecast`
to get more details about the forecasting operator.
"""
