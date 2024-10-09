#!/usr/bin/env python

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl.operator.lowcode.anomaly.const import NonTimeADSupportedModels
import yaml
import subprocess
import pandas as pd
import pytest
from time import sleep
from copy import deepcopy
import tempfile
import os
import numpy as np
from datetime import datetime
from ads.opctl.operator.cmd import run


MODELS = ["autots"]  # "automlx",

# Mandatory YAML parameters
TEMPLATE_YAML = {
    "kind": "operator",
    "type": "anomaly",
    "version": "v1",
    "spec": {
        "input_data": {
            "url": "https://raw.githubusercontent.com/facebook/prophet/c00f6a2d72229faa6acee8292bc01e14f16f599c/examples/example_retail_sales.csv",
        },
        "output_directory": {
            "url": "results",
        },
        "datetime_column": {
            "name": "ds",
        },
    },
}


DATASETS = [
    {
        "url": "https://raw.githubusercontent.com/numenta/NAB/master/data/artificialWithAnomaly/art_daily_flatmiddle.csv",
        "dt_col": "timestamp",
        "target": "value",
    },
]


parameters_short = []
for m in MODELS:
    for d in DATASETS:
        parameters_short.append((m, d))

MODELS = ["autots", "oneclasssvm", "isolationforest", "randomcutforest"]

@pytest.mark.parametrize("model", ["autots"])
def test_artificial_big(model):
    all_data = []
    TARGET_COLUMN = "sensor"
    TARGET_CATEGORY_COLUMN = "Meter ID"
    DATETIME_COLUMN = "Date"
    yr_in_30_min = pd.date_range(
        "2014-01-15 00:00:00", "2015-01-15 00:00:00", freq="30min"
    )

    for i in range(5):
        d1 = np.random.multivariate_normal(
            mean=np.array([-0.5, 0, 2]),
            cov=np.array([[1, 0, 0.5], [0, 1, 0.7], [0.5, 0.7, 1]]),
            size=len(yr_in_30_min),
        )
        df_i = pd.DataFrame(
            d1, columns=[TARGET_COLUMN, "extra reg 1", "extra reg 2"]
        )  # columns=[f"sensor {i}", f"extra reg {i}-1", f"extra reg {i}-2"]  # Uncomment for wide format
        if model not in NonTimeADSupportedModels.values():
            df_i[DATETIME_COLUMN] = yr_in_30_min
        df_i[TARGET_CATEGORY_COLUMN] = f"GHNI007894032{i}"
        all_data.append(df_i)

    d = pd.concat(all_data).reset_index(drop=True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        anomaly_yaml_filename = f"{tmpdirname}/anomaly.yaml"
        input_data = f"{tmpdirname}/data.csv"
        output_dirname = f"{tmpdirname}/results_{datetime.now()}"

        d.to_csv(input_data, index=False)

        yaml_i = deepcopy(TEMPLATE_YAML)
        yaml_i["spec"]["model"] = model
        yaml_i["spec"]["input_data"]["url"] = input_data
        yaml_i["spec"]["output_directory"]["url"] = output_dirname
        yaml_i["spec"]["target_column"] = TARGET_COLUMN
        yaml_i["spec"]["target_category_columns"] = [TARGET_CATEGORY_COLUMN]
        if model in NonTimeADSupportedModels.values():
            del yaml_i["spec"]["datetime_column"]
        else:
            yaml_i["spec"]["datetime_column"]["name"] = DATETIME_COLUMN

        # run(yaml_i, backend="operator.local", debug=False)

        with open(anomaly_yaml_filename, "w") as f:
            f.write(yaml.dump(yaml_i))
        sleep(0.1)
        subprocess.run(
            f"ads operator run -f {anomaly_yaml_filename} --debug", shell=True
        )
        sleep(0.1)
        subprocess.run(f"ls -a {output_dirname}/", shell=True)
        assert os.path.exists(f"{output_dirname}/report.html"), "Report not generated."


@pytest.mark.parametrize("model", MODELS)  # + ["auto"]
def test_artificial_small(model):
    # artificial data
    d1 = np.random.multivariate_normal(
        mean=np.array([-0.5, 0]), cov=np.array([[1, 0], [0, 1]]), size=100
    )
    d2 = np.random.multivariate_normal(
        mean=np.array([15, 10]), cov=np.array([[1, 0.3], [0.3, 1]]), size=100
    )
    outliers = np.array([[0, 10], [0, 9.5]])
    d = pd.DataFrame(
        np.concatenate([d1, d2, outliers], axis=0), columns=["val_1", "val_2"]
    )
    d = d.reset_index().rename({"index": "ds"}, axis=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        anomaly_yaml_filename = f"{tmpdirname}/anomaly.yaml"
        input_data = f"{tmpdirname}/data.csv"
        output_dirname = f"{tmpdirname}/results_{datetime.now()}"

        d.to_csv(input_data, index=False)

        yaml_i = deepcopy(TEMPLATE_YAML)
        yaml_i["spec"]["model"] = model
        yaml_i["spec"]["input_data"]["url"] = input_data
        yaml_i["spec"]["output_directory"]["url"] = output_dirname
        yaml_i["spec"]["contamination"] = 0.3
        if model in NonTimeADSupportedModels.values():
            del yaml_i["spec"]["datetime_column"]

        # run(yaml_i, debug=False)

        with open(anomaly_yaml_filename, "w") as f:
            f.write(yaml.dump(yaml_i))
        sleep(0.1)
        subprocess.run(
            f"ads operator run -f {anomaly_yaml_filename} --debug", shell=True
        )
        sleep(0.1)
        subprocess.run(f"ls -a {output_dirname}/", shell=True)
        assert os.path.exists(f"{output_dirname}/report.html"), "Report not generated."


@pytest.mark.parametrize("model", MODELS)
def test_validation(model):
    # artificial data
    d1 = np.random.multivariate_normal(
        mean=np.array([-0.5, 0]), cov=np.array([[1, 0], [0, 1]]), size=100
    )
    d2 = np.random.multivariate_normal(
        mean=np.array([15, 10]), cov=np.array([[1, 0.3], [0.3, 1]]), size=100
    )
    outliers = np.array([[0, 10], [0, 9.5]])
    d = pd.DataFrame(
        np.concatenate([d1, outliers, d2], axis=0), columns=["val_1", "val_2"]
    )
    anomaly_col = pd.DataFrame(
        np.concatenate([np.zeros(100), np.ones(2), np.zeros(100)], axis=0),
        columns=["anomaly"],
    )
    if model not in NonTimeADSupportedModels.values():
        d = d.reset_index().rename({"index": "ds"}, axis=1)
        anomaly_col["ds"] = d["ds"]
    v = d.copy()
    v["anomaly"] = anomaly_col["anomaly"]
    with tempfile.TemporaryDirectory() as tmpdirname:
        anomaly_yaml_filename = f"{tmpdirname}/anomaly.yaml"
        input_data = f"{tmpdirname}/data.csv"
        valid_data = f"{tmpdirname}/valid_data.csv"
        test_data = f"{tmpdirname}/test_data.csv"
        output_dirname = f"{tmpdirname}/results_{datetime.now()}"

        d.to_csv(input_data, index=False)
        v.to_csv(valid_data, index=False)
        anomaly_col.to_csv(test_data, index=False)

        yaml_i = deepcopy(TEMPLATE_YAML)
        yaml_i["spec"]["model"] = model
        yaml_i["spec"]["target_column"] = "val_1"
        yaml_i["spec"]["input_data"]["url"] = input_data
        yaml_i["spec"]["validation_data"] = {"url": valid_data}
        yaml_i["spec"]["test_data"] = {"url": test_data}
        yaml_i["spec"]["output_directory"]["url"] = output_dirname
        yaml_i["spec"]["contamination"] = 0.05
        if model in NonTimeADSupportedModels.values():
            del yaml_i["spec"]["datetime_column"]

        # run(yaml_i, backend="operator.local", debug=False)
        with open(anomaly_yaml_filename, "w") as f:
            f.write(yaml.dump(yaml_i))
        sleep(0.1)
        subprocess.run(
            f"ads operator run -f {anomaly_yaml_filename} --debug", shell=True
        )
        sleep(0.1)
        subprocess.run(f"ls -a {output_dirname}/", shell=True)
        assert os.path.exists(f"{output_dirname}/report.html"), "Report not generated."


@pytest.mark.parametrize("model, data_dict", parameters_short)
def test_load_datasets(model, data_dict):
    with tempfile.TemporaryDirectory() as tmpdirname:
        anomaly_yaml_filename = f"{tmpdirname}/anomaly.yaml"
        output_dirname = f"{tmpdirname}/results_{datetime.now()}"

        yaml_i = deepcopy(TEMPLATE_YAML)
        yaml_i["spec"]["model"] = model
        yaml_i["spec"]["input_data"]["url"] = data_dict["url"]
        if model in NonTimeADSupportedModels.values():
            del yaml_i["spec"]["datetime_column"]
        else:
            yaml_i["spec"]["datetime_column"]["name"] = data_dict["dt_col"]
        yaml_i["spec"]["output_directory"]["url"] = output_dirname

        # run(yaml_i, backend="operator.local", debug=False)

        with open(anomaly_yaml_filename, "w") as f:
            f.write(yaml.dump(yaml_i))
        sleep(0.5)
        subprocess.run(
            f"ads operator run -f {anomaly_yaml_filename} --debug", shell=True
        )
        sleep(0.1)
        subprocess.run(f"ls -a {output_dirname}/", shell=True)

        # train_metrics = pd.read_csv(f"{output_dirname}/metrics.csv")
        # print(train_metrics)
        # oultiers = pd.read_csv(f"{output_dirname}/anomaly.csv")
        # print(oultiers)
        assert os.path.exists(f"{output_dirname}/report.html"), "Report not generated."
