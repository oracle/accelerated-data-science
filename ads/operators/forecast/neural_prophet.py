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
# from prophet.plot import add_changepoints_to_plot
from neuralprophet import NeuralProphet
import pandas as pd
from ads.operators.forecast.utils import evaluate_metrics, _load_data, _clean_data

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def _preprocess_prophet(data, ds_column, datetime_format):
    data["ds"] = pd.to_datetime(data[ds_column], format=datetime_format)
    return data.drop([ds_column], axis=1)

def operate(operator):
    data = _load_data(operator.input_filename, operator.historical_data.get("format"), operator.storage_options, columns=operator.historical_data.get("columns"))
    data = _preprocess_prophet(data, operator.ds_column, operator.datetime_format)
    data = _clean_data(data)
    operator.data = data
    
    models = []
    outputs = []
    for i, col in enumerate(operator.target_columns):
        data_i = data[[col, "ds"]]
        print(f"using columns: {data_i.columns}")
        data_i.rename({col:"y"}, axis=1, inplace=True)
        
        model = NeuralProphet()
        model.fit(data_i)

        future = model.make_future_dataframe(df=data_i, periods=operator.horizon['periods']) #, freq=operator.horizon['interval_unit'])
        forecast = model.predict(future)

        print(f"-----------------Model {i}----------------------")
        print(forecast[['ds', 'yhat1']].tail())
        models.append(model)
        outputs.append(forecast)
    
    operator.models = models
    operator.outputs = outputs

    print("===========Done===========")
    output_total = pd.concat(operator.outputs)
    
    output_total.to_csv(operator.output_filename, storage_options=operator.storage_options)
    return operator
