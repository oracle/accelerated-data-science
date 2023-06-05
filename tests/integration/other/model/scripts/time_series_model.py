#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd
import statsmodels.api as sm
from ads.model.generic_model import GenericModel
from tests.integration.config import secrets


def timeseries():
    macrodata = sm.datasets.macrodata.load_pandas().data
    macrodata.index = pd.period_range("1959Q1", "2009Q3", freq="Q")
    y = macrodata["infl"] - macrodata["infl"].mean()
    y_pre = y.iloc[:-5]
    mod_pre = sm.tsa.arima.ARIMA(y_pre, order=(1, 0, 0), trend="n")
    res_pre = mod_pre.fit()
    y_update = y.iloc[-5:-3]
    res_post = res_pre.append(y_update)
    local_prediction = pd.concat([y_update, res_post.forecast("2009Q2")]).values

    return {
        "framework": GenericModel,
        "estimator": res_pre,
        "artifact_dir": "./artifact_folder/generic_model_ts",
        "inference_conda_env": f"oci://{secrets.other.BUCKET_1}@{secrets.common.NAMESPACE}/published_conda_environments/cpu/ads_env/1.0/ads_envv1_0",
        "inference_python_version": "3.7",
        "model_file_name": "model.pkl",
        "data": {
            "y_update": y_update.to_frame().to_json(),
            "x_test": "2009Q2",
            "start_period": "2008Q3",
            "end_period": "2008Q4",
        },
        "y_true": None,
        "local_pred": local_prediction,
        "score_py_path": "scripts/time_series_score.py",
    }
