#!/usr/bin/env python

# Copyright (c) 2023, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import json
import os
import subprocess
import tempfile
from copy import deepcopy
from time import sleep

import pandas as pd
import pytest
import yaml
import numpy as np

from ads.opctl.operator.cmd import run
from ads.opctl.operator.lowcode.forecast.__main__ import operate as forecast_operate
from ads.opctl.operator.lowcode.forecast.operator_config import ForecastOperatorConfig

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
    "lgbforecast",
    "xgbforecast",
    "theta",
    "ets",
    "auto-select",
    "auto-select-series",
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


def verify_explanations(tmpdirname, additional_cols, target_category_columns):
    result_files = os.listdir(f"{tmpdirname}/results")
    if model == "auto-select-series":
        # Find all local and global explanation files
        local_expl_files = [
            f
            for f in result_files
            if f.startswith("local_explanation_") and f.endswith(".csv")
        ]
        global_expl_files = [
            f
            for f in result_files
            if f.startswith("global_explanation_") and f.endswith(".csv")
        ]

        # Verify for each model's explanation files
        for loc_file, glb_file in zip(local_expl_files, global_expl_files):
            glb_expl = pd.read_csv(f"{tmpdirname}/results/{glb_file}", index_col=0)
            loc_expl = pd.read_csv(f"{tmpdirname}/results/{loc_file}")

            assert loc_expl.shape[0] == PERIODS
            columns = ["Date", "Series"]
            if not target_category_columns:
                columns.remove("Series")
            for x in columns:
                assert x in set(loc_expl.columns)
    else:
        glb_expl = pd.read_csv(
            f"{tmpdirname}/results/global_explanation.csv", index_col=0
        )
        loc_expl = pd.read_csv(f"{tmpdirname}/results/local_explanation.csv")

        assert loc_expl.shape[0] == PERIODS
        columns = ["Date", "Series"]
        if not target_category_columns:
            columns.remove("Series")
        for x in columns:
            assert x in set(loc_expl.columns)
    # for x in additional_cols:
    #     assert x in set(loc_expl.columns)
    #     assert x in set(glb_expl.index)


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
        if model == "auto-select":
            yaml_i["spec"]["model_kwargs"] = {
                "model_list": ["prophet", "xgbforecast", "ets"]
            }
            if dataset_name == f"{DATASET_PREFIX}dataset4.csv":
                pytest.skip("Skipping dataset4 with auto-select")  # todo:// ODSC-58584

        run(yaml_i, backend="operator.local", debug=False)
        subprocess.run(f"ls -a {output_data_path}", shell=True)
        if yaml_i["spec"]["generate_explanations"] and model not in [
            "automlx",
            "auto-select",
        ]:
            verify_explanations(
                tmpdirname=tmpdirname,
                additional_cols=additional_cols,
                target_category_columns=yaml_i["spec"]["target_category_columns"],
            )
        if include_test_data:
            result_files = os.listdir(f"{tmpdirname}/results")
            if model == "auto-select-series":
                # Find all metrics files for each model
                test_metrics_files = [
                    f
                    for f in result_files
                    if f.startswith("test_metrics_") and f.endswith(".csv")
                ]
                train_metrics_files = [
                    f
                    for f in result_files
                    if f.startswith("metrics_") and f.endswith(".csv")
                ]

                # Print metrics for each model
                for test_file, train_file in zip(
                    test_metrics_files, train_metrics_files
                ):
                    print(
                        f"\nMetrics for {test_file.replace('test_metrics_', '').replace('.csv', '')}:"
                    )
                    test_metrics = pd.read_csv(f"{tmpdirname}/results/{test_file}")
                    print("Test metrics:")
                    print(test_metrics)
                    train_metrics = pd.read_csv(f"{tmpdirname}/results/{train_file}")
                    print("Train metrics:")
                    print(train_metrics)
            else:
                test_metrics = pd.read_csv(f"{tmpdirname}/results/test_metrics.csv")
                print(test_metrics)
                train_metrics = pd.read_csv(f"{tmpdirname}/results/metrics.csv")
                print(train_metrics)


@pytest.mark.parametrize("model", MODELS[:-2])
def test_pandas_to_historical(model):
    df = pd.read_csv(f"{DATASET_PREFIX}dataset1.csv")

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_data_path = f"{tmpdirname}/results"
        yaml_i = deepcopy(TEMPLATE_YAML)
        yaml_i["spec"]["model"] = model
        yaml_i["spec"]["historical_data"].pop("url")
        yaml_i["spec"]["historical_data"]["data"] = df
        yaml_i["spec"]["target_column"] = "Y"
        yaml_i["spec"]["datetime_column"]["name"] = DATETIME_COL
        yaml_i["spec"]["horizon"] = PERIODS
        yaml_i["spec"]["output_directory"]["url"] = output_data_path
        if model == "automlx":
            yaml_i["spec"]["model_kwargs"] = {"time_budget": 2}
        operator_config = ForecastOperatorConfig.from_dict(yaml_i)
        forecast_operate(operator_config)
        check_output_for_errors(output_data_path)


@pytest.mark.parametrize("model", ["prophet", "neuralprophet"])
def test_pandas_to_historical_test(model):
    df = pd.read_csv(f"{DATASET_PREFIX}dataset4.csv")
    df_train = df[:-PERIODS]
    df_test = df[-PERIODS:]

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_data_path = f"{tmpdirname}/results"
        yaml_i = deepcopy(TEMPLATE_YAML)
        yaml_i["spec"]["model"] = model
        yaml_i["spec"]["historical_data"].pop("url")
        yaml_i["spec"]["historical_data"]["data"] = df_train
        yaml_i["spec"]["test_data"] = {"data": df_test}
        yaml_i["spec"]["target_column"] = "Y"
        yaml_i["spec"]["datetime_column"]["name"] = DATETIME_COL
        yaml_i["spec"]["horizon"] = PERIODS
        yaml_i["spec"]["output_directory"]["url"] = output_data_path
        if model == "automlx":
            yaml_i["spec"]["model_kwargs"] = {"time_budget": 2}
        operator_config = ForecastOperatorConfig.from_dict(yaml_i)
        forecast_operate(operator_config)
        check_output_for_errors(output_data_path)
        test_metrics = pd.read_csv(f"{output_data_path}/metrics.csv")
        print(test_metrics)


# CostAD
@pytest.mark.parametrize("model", ["prophet", "neuralprophet"])
def test_pandas_to_historical_test2(model):
    df = pd.read_csv(f"{DATASET_PREFIX}dataset5.csv")
    df_train = df[:-1]
    df_test = df[-1:]
    df1, df2 = None, None

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_data_path = f"{tmpdirname}/results"
        yaml_i = deepcopy(TEMPLATE_YAML)
        yaml_i["spec"]["model"] = model
        yaml_i["spec"]["historical_data"].pop("url")
        yaml_i["spec"]["historical_data"]["data"] = df_train
        yaml_i["spec"]["test_data"] = {"data": df_test}
        yaml_i["spec"]["target_column"] = "Y"
        yaml_i["spec"]["datetime_column"]["name"] = DATETIME_COL
        yaml_i["spec"]["datetime_column"]["format"] = "%d/%m/%Y"
        yaml_i["spec"]["horizon"] = 1
        yaml_i["spec"]["output_directory"]["url"] = output_data_path
        if model == "automlx":
            yaml_i["spec"]["model_kwargs"] = {"time_budget": 2}
        operator_config = ForecastOperatorConfig.from_dict(yaml_i)
        results = forecast_operate(operator_config)
        # check_output_for_errors(output_data_path)
        test_metrics = pd.read_csv(f"{output_data_path}/metrics.csv")
        df1 = results.get_test_metrics()
        df2 = results.get_forecast()


def check_output_for_errors(output_data_path):
    # try:
    # List files in the directory
    result = subprocess.run(
        f"ls -a {output_data_path}",
        shell=True,
        check=True,
        text=True,
        capture_output=True,
    )
    files = result.stdout.splitlines()

    # Check if errors.json is in the directory
    if "errors.json" in files:
        errors_file_path = os.path.join(output_data_path, "errors.json")

        # Read the errors.json file
        with open(errors_file_path, "r") as f:
            errors_content = json.load(f)

        # Extract and raise the error message
        # error_message = errors_content.get("message", "An error occurred.")
        raise Exception(errors_content)

    print("No errors.json file found. Directory is clear.")

    # except subprocess.CalledProcessError as e:
    #     print(f"Error listing files in directory: {e}")
    # except FileNotFoundError:
    #     print("The directory does not exist.")
    # except json.JSONDecodeError:
    #     print("errors.json is not a valid JSON file.")
    # except Exception as e:
    #     print(f"Raised error: {e}")


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

        test_metrics = pd.read_csv(f"{tmpdirname}/results/metrics.csv")
        print(test_metrics)
        train_metrics = pd.read_csv(f"{tmpdirname}/results/train_metrics.csv")
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


def test_missing_data_autoselect_series():
    """Test case for auto-select-series with missing data."""
    data = {
        "Date": pd.to_datetime(
            [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
                "2023-01-06",
                "2023-01-07",
                "2023-01-08",
                "2023-01-09",
                "2023-01-10",
            ]
        ),
        "Y": [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
        "Category": ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A"],
    }
    df = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_data_path = f"{tmpdirname}/results"
        yaml_i = deepcopy(TEMPLATE_YAML)
        yaml_i["spec"]["model"] = "auto-select-series"
        yaml_i["spec"]["historical_data"].pop("url")
        yaml_i["spec"]["historical_data"]["data"] = df
        yaml_i["spec"]["target_column"] = "Y"
        yaml_i["spec"]["datetime_column"]["name"] = "Date"
        yaml_i["spec"]["target_category_columns"] = ["Category"]
        yaml_i["spec"]["horizon"] = 2
        yaml_i["spec"]["output_directory"]["url"] = output_data_path

        operator_config = ForecastOperatorConfig.from_dict(yaml_i)
        forecast_operate(operator_config)
        check_output_for_errors(output_data_path)


if __name__ == "__main__":
    pass
