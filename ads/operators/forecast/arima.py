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
from ads.operators.forecast.utils import load_data_dict, _write_data

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def operate(operator):
    operator = load_data_dict(operator)
    full_data_dict = operator.full_data_dict

    models = []
    outputs = []
    for i, (target, df) in enumerate(full_data_dict.items()):
        df[operator.ds_column] = pd.to_datetime(
            df[operator.ds_column], format=operator.datetime_format
        )
        df = df.set_index(operator.ds_column)

        data_i = df[df[target].notna()]

        additional_regressors = set(data_i.columns) - {target, operator.ds_column}
        print(f"Additional Regressors Detected {list(additional_regressors)}")

        y = data_i[target]
        X_in = None
        if len(additional_regressors):
            X_in = data_i.drop(target, axis=1)

        print(f"y: {y}, X_in: {X_in}")

        model = pm.auto_arima(y=y, X=X_in)

        # # TODO: Add support for interval in horizon
        # # TODO: Add support for model kwargs
        start_date = y.index.values[-1]
        n_periods = operator.horizon.get("periods")
        interval_unit = operator.horizon.get("interval_unit")
        if len(additional_regressors):
            X = df[df[target].isnull()].drop(target, axis=1)
        else:
            X = pd.date_range(start=start_date, periods=n_periods, freq=interval_unit)

        yhat, conf_int = model.predict(
            n_periods=n_periods, X=X, return_conf_int=True, alpha=0.05
        )
        # yhat.index = X[operator.ds_column]
        yhat_clean = pd.DataFrame(yhat, index=yhat.index, columns=["yhat"])
        conf_int_clean = pd.DataFrame(
            conf_int, index=yhat.index, columns=["yhat_lower", "yhat_upper"]
        )
        forecast = pd.concat([yhat_clean, conf_int_clean], axis=1)
        print(f"-----------------Model {i}----------------------")
        print(forecast[["yhat", "yhat_lower", "yhat_upper"]].tail())
        models.append(model)
        outputs.append(forecast)

    operator.models = models
    operator.outputs = outputs

    print("===========Done===========")
    outputs_merged = outputs.copy()
    for i, col in enumerate(operator.target_columns):
        outputs_merged[i] = outputs_merged[i].rename(
            lambda x: x + "_" + col if x != "ds" else x, axis=1
        )
    output_total = pd.concat(outputs_merged, axis=1)
    _write_data(output_total, operator.output_filename, "csv", operator.storage_options)

    # data_merged = operator.original_user_data.join(operator.original_additional_data)
    data_merged = pd.concat(
        [
            v[v[k].notna()].set_index(operator.ds_column)
            for k, v in full_data_dict.items()
        ],
        axis=1,
    ).reset_index()
    return data_merged, models, outputs
