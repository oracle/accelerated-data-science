#!/usr/bin/env python

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import os
import yaml
import tempfile
import subprocess
import pandas as pd
import pytest
from time import sleep, time
from copy import deepcopy
from pathlib import Path
import random
import pathlib
import datetime
from ads.opctl.operator.cmd import run


DATASET_PREFIX = f"{os.path.dirname(os.path.abspath(__file__))}/../data/timeseries/"

DATASETS_LIST = [
    {"filename": f"{DATASET_PREFIX}dataset1.csv"},
    {"filename": f"{DATASET_PREFIX}dataset2.csv"},
    {"filename": f"{DATASET_PREFIX}dataset3.csv"},
    {"filename": f"{DATASET_PREFIX}dataset4.csv", "include_test_data": False},
]

MODELS = [
    "arima",
    "automlx",
    "prophet",
    "neuralprophet",
    "autots",
    # "auto",
]

TEMPLATE_YAML = {
    "kind": "operator",
    "type": "forecast",
    "version": "v1",
    "spec": {
        "historical_data": {
            "url": None,
        },
        "output_directory": {
            "url": "results",
        },
        "model": None,
        "target_column": None,
        "datetime_column": {
            "name": None,
        },
        "target_category_columns": [],
        "horizon": None,
        "generate_explanations": False,
    },
}


PERIODS = 5
MAX_ADDITIONAL_COLS = 3
SAMPLE_FRACTION = 1

DATETIME_COL = "Date"

parameters_short = []

for dataset_i in DATASETS_LIST:  #  + [DATASETS_LIST[-2]]
    for model in MODELS:
        if model != "automlx" and dataset_i != f"{DATASET_PREFIX}dataset3.csv":
            parameters_short.append((model, dataset_i))


def verify_explanations(tmpdirname, additional_cols):
    glb_expl = pd.read_csv(f"{tmpdirname}/results/global_explanation.csv", index_col=0)
    loc_expl = pd.read_csv(f"{tmpdirname}/results/local_explanation.csv")
    assert loc_expl.shape[0] == PERIODS
    for x in ["Date", "Series"]:
        assert x in set(loc_expl.columns)
    # for x in additional_cols:
    #     assert x in set(loc_expl.columns)
    #     assert x in set(glb_expl.index)
    assert "Series 1" in set(glb_expl.columns)


@pytest.mark.parametrize("model, data_details", parameters_short)
def test_load_datasets(model, data_details):
    dataset_name = data_details["filename"]
    target = data_details.get("target", "Y")
    dt_format = data_details.get("format")
    include_test_data = data_details.get("include_test_data", True)

    dataset_i = pd.read_csv(dataset_name)
    additional_cols = list(set(dataset_i.columns) - {DATETIME_COL, target})

    print(dataset_name, len(target))
    with tempfile.TemporaryDirectory() as tmpdirname:
        historical_data_path = f"{tmpdirname}/primary_data.csv"
        additional_data_path = f"{tmpdirname}/add_data.csv"
        test_data_path = f"{tmpdirname}/test_data.csv"
        output_data_path = f"{tmpdirname}/results"
        yaml_i = deepcopy(TEMPLATE_YAML)

        dataset_i[[DATETIME_COL, target]][:-PERIODS].to_csv(
            historical_data_path, index=False
        )
        dataset_i[[DATETIME_COL, target]][-PERIODS:].to_csv(test_data_path, index=False)

        if len(additional_cols) > 0:
            if len(additional_cols) > MAX_ADDITIONAL_COLS:
                selected_add_cols = [DATETIME_COL] + additional_cols[
                    :MAX_ADDITIONAL_COLS
                ]
            additional_data = dataset_i[selected_add_cols]
            additional_data.to_csv(additional_data_path, index=False)
            yaml_i["spec"]["additional_data"] = {"url": additional_data_path}

        yaml_i["spec"]["historical_data"]["url"] = historical_data_path
        if include_test_data:
            yaml_i["spec"]["test_data"] = {"url": test_data_path}
        yaml_i["spec"]["output_directory"]["url"] = output_data_path
        yaml_i["spec"]["model"] = model
        yaml_i["spec"]["target_column"] = target
        yaml_i["spec"]["datetime_column"]["name"] = DATETIME_COL
        if dt_format:
            yaml_i["spec"]["datetime_column"]["format"] = dt_format
        yaml_i["spec"]["horizon"] = PERIODS
        yaml_i["spec"]["generate_metrics"] = True
        if yaml_i["spec"].get("additional_data") is not None and model != "autots":
            yaml_i["spec"]["generate_explanations"] = True
        if model == "autots":
            yaml_i["spec"]["model_kwargs"] = {"model_list": "superfast"}
        if model == "automlx":
            yaml_i["spec"]["model_kwargs"] = {"time_budget": 2}

        run(yaml_i, backend="operator.local", debug=False)
        subprocess.run(f"ls -a {output_data_path}", shell=True)
        if yaml_i["spec"]["generate_explanations"] and model != "automlx":
            verify_explanations(
                tmpdirname=tmpdirname,
                additional_cols=additional_cols,
            )
        if include_test_data:
            test_metrics = pd.read_csv(f"{tmpdirname}/results/test_metrics.csv")
            print(test_metrics)
            train_metrics = pd.read_csv(f"{tmpdirname}/results/metrics.csv")
            print(train_metrics)


def run_operator(
    historical_data_path,
    additional_data_path,
    test_data_path,
    generate_train_metrics=True,
    output_data_path=None,
):
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_data_path = output_data_path or f"{tmpdirname}/results"
        yaml_i = deepcopy(TEMPLATE_YAML)
        generate_train_metrics = True

        yaml_i["spec"]["additional_data"] = {"url": additional_data_path}
        yaml_i["spec"]["historical_data"]["url"] = historical_data_path
        yaml_i["spec"]["test_data"] = {"url": test_data_path}
        yaml_i["spec"]["output_directory"]["url"] = output_data_path
        yaml_i["spec"]["model"] = model
        yaml_i["spec"]["target_column"] = "Sales"
        yaml_i["spec"]["datetime_column"]["name"] = DATETIME_COL
        yaml_i["spec"]["target_category_columns"] = ["Store"]
        yaml_i["spec"]["horizon"] = PERIODS

        if generate_train_metrics:
            yaml_i["spec"]["generate_metrics"] = generate_train_metrics
        if model == "autots":
            yaml_i["spec"]["model_kwargs"] = {"model_list": "superfast"}
        if model == "automlx":
            yaml_i["spec"]["model_kwargs"] = {"time_budget": 2}

        forecast_yaml_filename = f"{tmpdirname}/forecast.yaml"
        with open(f"{tmpdirname}/forecast.yaml", "w") as f:
            f.write(yaml.dump(yaml_i))
        sleep(0.1)
        subprocess.run(f"ads operator run -f {forecast_yaml_filename}", shell=True)
        sleep(0.1)
        subprocess.run(f"ls -a {output_data_path}", shell=True)

        test_metrics = pd.read_csv(f"{tmpdirname}/results/test_metrics.csv")
        print(test_metrics)
        train_metrics = pd.read_csv(f"{tmpdirname}/results/metrics.csv")
        print(train_metrics)


# parameters_datetime = []
# DATETIME_FORMATS_TO_TEST = [
#     ["%Y", datetime.timedelta()],
#     ["%y", ],
#     ["%b-%d-%Y",],
#     ["%d-%m-%y",],
#     ["%d/%m/%y %H:%M:%S",],
# ]

# for dt_format in DATETIME_FORMATS_TO_TEST:
#     for (model) in MODELS:
#         parameters_datetime.append((model, dt_format))


# @pytest.mark.parametrize("model, dt_format", parameters_datetime)
# def test_datetime_formats(model=model, dt_format=dt_format):
#     curr_dir = pathlib.Path(__file__).parent.resolve()
#     data_folder = f"{curr_dir}/../data/"
#     np.arrange((1000, 12))
#     d1 = np.random.multivariate_normal(
#         mean=np.array([-0.5, 0, 2]),
#         cov=np.array([[1, 0, 0.5], [0, 1, 0.7], [0.5, 0.7, 1]]),
#         size=len,
#     )
#     now = datetime.datetime.now()
#     now_formatted = now.strftime(dt_format)

#     historical_data_path = f"{data_folder}/rs_10_prim.csv"
#     additional_data_path = f"{data_folder}/rs_10_add.csv"
#     test_data_path = f"{data_folder}/rs_10_test.csv"

#     with tempfile.TemporaryDirectory() as tmpdirname:
#         output_data_path = f"{tmpdirname}/results"
#         yaml_i = deepcopy(TEMPLATE_YAML)
#         generate_train_metrics = True


if __name__ == "__main__":
    pass
