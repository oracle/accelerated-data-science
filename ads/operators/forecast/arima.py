#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import logging
import os
import sys
from ads.jobs.builders.runtimes.python_runtime import PythonRuntime
import datapane as dp
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import pandas as pd
from ads.operators.forecast.utils import evaluate_metrics, _load_data, _clean_data, _write_data

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def _preprocess_arima(data, ds_column, datetime_format):
    data[ds_column] = pd.to_datetime(data[ds_column], format=datetime_format)
    data = data.set_index(ds_column)
    return data

def operate(operator):
    data = _load_data(operator.input_filename, operator.historical_data.get("format"), operator.storage_options, columns=operator.historical_data.get("columns"))
    data = _preprocess_arima(data, operator.ds_column, operator.datetime_format)
    data = _clean_data(data)
    operator.data = data
    
    models = []
    outputs = []
    for i, col in enumerate(operator.target_columns):
        data_i = data[col]
        
        model = pm.auto_arima(data_i)
        start_date = data_i.index.values[-1]
        n_periods = operator.horizon.get("periods")
        interval_unit = operator.horizon.get("interval_unit")
        # TODO: Add support for interval in horizon
        # TODO: Add support for model kwargs
        X = pd.date_range(start=start_date, periods=n_periods, freq=interval_unit)
        yhat, conf_int = model.predict(n_periods=n_periods, X=X, return_conf_int=True, alpha=0.05)
        yhat.index = X
        yhat_clean = pd.DataFrame(yhat, index=X, columns=["yhat"])
        conf_int_clean = pd.DataFrame(conf_int, index=X, columns=['yhat_lower', 'yhat_upper'])
        forecast = pd.concat([yhat_clean, conf_int_clean], axis=1)
        print(f"-----------------Model {i}----------------------")
        print(forecast[['yhat', 'yhat_lower', 'yhat_upper']].tail())
        forecast_output = forecast.rename(lambda x: x+"_"+col if x != 'ds' else x, axis=1)
        models.append(model)
        outputs.append(forecast_output)
    
    operator.models = models
    operator.outputs = outputs

    print("===========Done===========")
    output_total = pd.concat(operator.outputs, axis=1)
    
    _write_data(output_total, operator.output_filename, "csv", operator.storage_options)
    return data, models, outputs
