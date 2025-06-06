#!/usr/bin/env python
from unittest.mock import patch

# Copyright (c) 2023, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import yaml
import tempfile
import subprocess
import pandas as pd
import numpy as np
import pytest
from time import sleep, time
from copy import deepcopy
from pathlib import Path
import random
import pathlib
import datetime
from ads.opctl.operator.lowcode.common.errors import (
    InputDataError,
    InvalidParameterError,
    PermissionsError,
    DataMismatchError,
)
from ads.opctl.operator.lowcode.forecast.errors import (
    ForecastSchemaYamlError,
    ForecastInputDataError,
)
from ads.opctl.operator.lowcode.forecast.operator_config import ForecastOperatorConfig

from ads.opctl.operator.lowcode.forecast.utils import smape
from ads.opctl.operator.cmd import run
from ads.opctl.operator.lowcode.forecast.__main__ import operate
import os
import json
import math

NUM_ROWS = 1000
NUM_SERIES = 10
HORIZON = 5

HISTORICAL_DATETIME_COL = pd.Series(
    np.concatenate(
        [
            datetime.datetime.strptime("10/12/22 22:45:59", "%d/%m/%y %H:%M:%S")
            + np.arange(NUM_ROWS - HORIZON) * datetime.timedelta(hours=1)
        ]
        * NUM_SERIES
    ),
    name="Date",
)
ADDITIONAL_DATETIME_COL = pd.Series(
    np.concatenate(
        [
            datetime.datetime.strptime("10/12/22 22:45:59", "%d/%m/%y %H:%M:%S")
            + np.arange(NUM_ROWS) * datetime.timedelta(hours=1)
        ]
        * NUM_SERIES
    ),
    name="Date",
)
TEST_DATETIME_COL = pd.Series(
    np.concatenate(
        [ADDITIONAL_DATETIME_COL[NUM_ROWS - HORIZON : NUM_ROWS]] * NUM_SERIES
    ),
    name="Date",
)

BASE_DATA = np.random.multivariate_normal(
    mean=np.array([-0.5, 0, 2]),
    cov=np.array([[1, 0, 0.5], [0, 1, 0.7], [0.5, 0.7, 1]]),
    size=NUM_ROWS * NUM_SERIES,
)

TARGET_COL = pd.Series(
    np.concatenate(
        [
            BASE_DATA[i : i + NUM_ROWS - HORIZON, 2]
            for i in range(0, NUM_ROWS * NUM_SERIES, NUM_ROWS)
        ]
    ),
    name="Sales",
)
TEST_TARGET_COL = pd.Series(
    np.concatenate(
        [
            BASE_DATA[i - HORIZON : i, 2]
            for i in range(NUM_ROWS, NUM_ROWS * NUM_SERIES + 1, NUM_ROWS)
        ]
    ),
    name="Sales",
)

ADD_COLS = pd.DataFrame(BASE_DATA[:, :2], columns=["var1", "var2"])

ADDITIONAL_SERIES_COL = pd.Series(
    np.concatenate([[i] * NUM_ROWS for i in range(NUM_SERIES)]), name="Store"
)
HISTORICAL_SERIES_COL = pd.Series(
    np.concatenate([[i] * (NUM_ROWS - HORIZON) for i in range(NUM_SERIES)]),
    name="Store",
)
TEST_SERIES_COL = pd.Series(
    np.concatenate([[i] * (HORIZON) for i in range(NUM_SERIES)]), name="Store"
)

CONST_COL = pd.Series(np.ones(NUM_ROWS * NUM_SERIES), name="const")

ORDINAL_COL = pd.Series(np.arange(NUM_ROWS * NUM_SERIES), name="ordinal")

HISTORICAL_DATA = pd.concat(
    [
        HISTORICAL_DATETIME_COL,
        HISTORICAL_SERIES_COL,
        TARGET_COL,
    ],
    axis=1,
)

ADDITIONAL_DATA = pd.concat(
    [
        ADDITIONAL_DATETIME_COL,
        ADDITIONAL_SERIES_COL,
        ADD_COLS,
        CONST_COL,
        ORDINAL_COL,
    ],
    axis=1,
)

TEST_DATA = pd.concat(
    [
        TEST_DATETIME_COL,
        TEST_SERIES_COL,
        TEST_TARGET_COL,
    ],
    axis=1,
)

MODELS = [
    "arima",
    "automlx",
    "prophet",
    "neuralprophet",
    "autots",
    # "lgbforecast",
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


@pytest.fixture(autouse=True)
def operator_setup():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


def run_yaml(tmpdirname, yaml_i, output_data_path, test_metrics_check=True):
    run(yaml_i, backend="operator.local", debug=True)
    subprocess.run(f"ls -a {output_data_path}", shell=True)

    if test_metrics_check:
        test_metrics = pd.read_csv(f"{output_data_path}/test_metrics.csv")
        print(test_metrics)
    train_metrics = pd.read_csv(f"{output_data_path}/metrics.csv")
    print(train_metrics)


def populate_yaml(
    tmpdirname=None,
    model="auto-select",
    historical_data_path=None,
    additional_data_path=None,
    test_data_path=None,
    output_data_path=None,
    preprocessing=None,
):
    if historical_data_path is None:
        historical_data_path, additional_data_path, test_data_path = setup_rossman()
    assert tmpdirname is not None, "Error casued from incomplete setup function"

    output_data_path = output_data_path or f"{tmpdirname}/results"
    yaml_i = deepcopy(TEMPLATE_YAML)
    generate_train_metrics = True

    yaml_i["spec"]["additional_data"] = {"url": additional_data_path}
    yaml_i["spec"]["historical_data"]["url"] = historical_data_path
    if test_data_path is not None:
        yaml_i["spec"]["test_data"] = {"url": test_data_path}
    yaml_i["spec"]["output_directory"]["url"] = output_data_path
    yaml_i["spec"]["model"] = model
    yaml_i["spec"]["target_column"] = "Sales"
    yaml_i["spec"]["datetime_column"]["name"] = "Date"
    yaml_i["spec"]["target_category_columns"] = ["Store"]
    yaml_i["spec"]["horizon"] = HORIZON
    if preprocessing:
        yaml_i["spec"]["preprocessing"] = preprocessing
    if generate_train_metrics:
        yaml_i["spec"]["generate_metrics"] = generate_train_metrics
    if model == "autots":
        yaml_i["spec"]["model_kwargs"] = {"model_list": "superfast"}
    if model == "automlx":
        yaml_i["spec"]["model_kwargs"] = {"time_budget": 2}

    return yaml_i, output_data_path


def run_operator(
    tmpdirname=None,
    model="auto-select",
    historical_data_path=None,
    additional_data_path=None,
    test_data_path=None,
    output_data_path=None,
):
    yaml_i, output_data_path = populate_yaml(
        tmpdirname=tmpdirname,
        model=model,
        historical_data_path=historical_data_path,
        additional_data_path=additional_data_path,
        test_data_path=test_data_path,
        output_data_path=output_data_path,
    )
    run_yaml(tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path)


def setup_rossman():
    curr_dir = pathlib.Path(__file__).parent.resolve()
    data_folder = f"{curr_dir}/../data/"
    historical_data_path = f"{data_folder}/rs_10_prim.csv"
    additional_data_path = f"{data_folder}/rs_10_add.csv"
    test_data_path = f"{data_folder}/rs_10_test.csv"
    return historical_data_path, additional_data_path, test_data_path


def setup_faulty_rossman():
    curr_dir = pathlib.Path(__file__).parent.resolve()
    data_folder = f"{curr_dir}/../data"
    historical_data_path = f"{data_folder}/rs_2_prim.csv"
    additional_data_path = f"{data_folder}/rs_2_add_encoded.csv"
    return historical_data_path, additional_data_path


def setup_small_rossman():
    curr_dir = pathlib.Path(__file__).parent.resolve()
    data_folder = f"{curr_dir}/../data/"
    historical_data_path = f"{data_folder}/rs_1_prim.csv"
    additional_data_path = f"{data_folder}/rs_1_add.csv"
    return historical_data_path, additional_data_path


def setup_artificial_data(tmpdirname, hist_data=None, add_data=None, test_data=None):
    if hist_data is None:
        hist_data = HISTORICAL_DATA
    if add_data is None:
        add_data = ADDITIONAL_DATA
    if test_data is None:
        test_data = TEST_DATA

    historical_data_path = f"{tmpdirname}/data.csv"
    hist_data.to_csv(historical_data_path, index=False)

    additional_data_path = f"{tmpdirname}/add_data.csv"
    add_data.to_csv(additional_data_path, index=False)

    test_data_path = f"{tmpdirname}/test_data.csv"
    test_data.to_csv(test_data_path, index=False)

    return historical_data_path, additional_data_path, test_data_path


@pytest.mark.parametrize("model", ["prophet"])
def test_rossman(operator_setup, model):
    run_operator(
        tmpdirname=operator_setup,
        model=model,
    )


@pytest.mark.parametrize("model", ["prophet"])
def test_historical_data(operator_setup, model):
    tmpdirname = operator_setup
    historical_data_path, additional_data_path, _ = setup_artificial_data(tmpdirname)
    yaml_i, output_data_path = populate_yaml(
        tmpdirname=tmpdirname,
        historical_data_path=historical_data_path,
        additional_data_path=additional_data_path,
    )

    # Test historical data missing error
    historical_data = yaml_i["spec"]["historical_data"]
    yaml_i["spec"]["historical_data"]["url"] = None
    with pytest.raises(InvalidParameterError):
        run_yaml(
            tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path
        )

    yaml_i["spec"].pop("historical_data")
    yaml_i["spec"]["TEST"] = historical_data
    with pytest.raises(InvalidParameterError):
        run_yaml(
            tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path
        )

    yaml_i["spec"]["historical_data"] = historical_data
    with pytest.raises(InvalidParameterError):
        run_yaml(
            tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path
        )
    yaml_i["spec"].pop("TEST")

    # Test incrrect target column name error
    yaml_i["spec"]["target_column"] = "TEST"
    with pytest.raises(InvalidParameterError):
        run_yaml(
            tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path
        )
    yaml_i["spec"]["target_column"] = "Sales"

    # Test incorrect series column name error
    yaml_i["spec"]["target_category_column"] = ["TEST"]
    with pytest.raises(InvalidParameterError):
        run_yaml(
            tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path
        )
    yaml_i["spec"]["target_category_column"] = ["Store"]

    # Test incorrect datetime column name error
    yaml_i["spec"]["datetime_column"] = {"name": "TEST"}
    with pytest.raises(InvalidParameterError):
        run_yaml(
            tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path
        )
    yaml_i["spec"]["datetime_column"] = {"name": "Date", "format": "TEST"}
    with pytest.raises(InvalidParameterError):
        run_yaml(
            tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path
        )


@pytest.mark.parametrize("model", MODELS)
def test_0_series(operator_setup, model):
    tmpdirname = operator_setup
    hist_data_0 = pd.concat(
        [
            HISTORICAL_DATETIME_COL[: NUM_ROWS - HORIZON],
            TARGET_COL[: NUM_ROWS - HORIZON],
        ],
        axis=1,
    )
    add_data_0 = pd.concat(
        [
            ADDITIONAL_DATETIME_COL[:NUM_ROWS],
            ADD_COLS[:NUM_ROWS],
            CONST_COL[:NUM_ROWS],
            ORDINAL_COL[:NUM_ROWS],
        ],
        axis=1,
    )
    test_data_0 = pd.concat(
        [
            TEST_DATETIME_COL[:HORIZON],
            TEST_TARGET_COL[:HORIZON],
        ],
        axis=1,
    )

    print(
        f"hist_data_0: {hist_data_0}\nadd_data_0:{add_data_0}\ntest_data_0:{test_data_0}\n"
    )

    historical_data_path, additional_data_path, test_data_path = setup_artificial_data(
        tmpdirname, hist_data_0, add_data_0, test_data_0
    )
    yaml_i, output_data_path = populate_yaml(
        tmpdirname=tmpdirname,
        model=model,
        historical_data_path=historical_data_path,
        additional_data_path=additional_data_path,
        test_data_path=test_data_path,
        preprocessing={"enabled": False},
    )
    with pytest.raises(DataMismatchError):
        run_yaml(
            tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path
        )
    yaml_i["spec"].pop("target_category_columns")
    yaml_i["spec"]["generate_explanations"] = True
    run_yaml(tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path)
    output_files = [
        "forecast.csv",
        "metrics.csv",
        "test_metrics.csv",
        "report.html",
        "local_explanation.csv",
        "global_explanation.csv",
    ]
    if model == "autots":
        # explanations are not supported for autots
        output_files.remove("local_explanation.csv")
        output_files.remove("global_explanation.csv")
    for file in output_files:
        file_path = os.path.join(output_data_path, file)
        with open(file_path, "r", encoding="utf-8") as cur_file:
            content = cur_file.read()
            assert "Series 1" not in content, f"'Series 1' found in file: {file}"
    yaml_i["spec"].pop("additional_data")
    yaml_i["spec"].pop("generate_explanations")
    run_yaml(tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path)
    yaml_i["spec"].pop("test_data")
    run_yaml(tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path)
    # Todo test horizon mismatch with add data and/or test data


@pytest.mark.parametrize("model", MODELS)
def test_add_data_mismatch(operator_setup, model):
    tmpdirname = operator_setup
    historical_data_path, additional_data_path, _ = setup_artificial_data(tmpdirname)
    yaml_i, output_data_path = populate_yaml(
        tmpdirname=tmpdirname,
        historical_data_path=historical_data_path,
        additional_data_path=additional_data_path,
    )
    with pytest.raises(DataMismatchError):
        yaml_i["spec"]["horizon"] = HORIZON - 1
        run_yaml(
            tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path
        )
    with pytest.raises(DataMismatchError):
        yaml_i["spec"]["horizon"] = HORIZON + 1
        run_yaml(
            tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path
        )


@pytest.mark.parametrize("model", MODELS)
def test_invalid_dates(operator_setup, model):
    tmpdirname = operator_setup
    hist_data = HISTORICAL_DATA[:100]
    hist_data = pd.concat([hist_data, hist_data])
    add_data = ADDITIONAL_DATA[:100]
    add_data = pd.concat([add_data, add_data])

    historical_data_path, additional_data_path, _ = setup_artificial_data(
        tmpdirname, hist_data=hist_data, add_data=add_data
    )
    yaml_i, output_data_path = populate_yaml(
        tmpdirname=tmpdirname,
        historical_data_path=historical_data_path,
        additional_data_path=additional_data_path,
    )
    with pytest.raises(DataMismatchError):
        run_yaml(
            tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path
        )


# def test_disabling_outlier_treatment(operator_setup):
#     tmpdirname = operator_setup
#     NUM_ROWS = 100
#     hist_data_0 = pd.concat(
#         [
#             HISTORICAL_DATETIME_COL[: NUM_ROWS - HORIZON],
#             TARGET_COL[: NUM_ROWS - HORIZON],
#         ],
#         axis=1,
#     )
#     outliers = [1000, -800]
#     hist_data_0.at[40, "Sales"] = outliers[0]
#     hist_data_0.at[75, "Sales"] = outliers[1]
#     historical_data_path, additional_data_path, test_data_path = setup_artificial_data(
#         tmpdirname, hist_data_0
#     )

#     yaml_i, output_data_path = populate_yaml(
#         tmpdirname=tmpdirname,
#         model="arima",
#         historical_data_path=historical_data_path,
#     )
#     yaml_i["spec"].pop("target_category_columns")
#     yaml_i["spec"].pop("additional_data")

#     # running default pipeline where outlier will be treated
#     run_yaml(
#         tmpdirname=tmpdirname,
#         yaml_i=yaml_i,
#         output_data_path=output_data_path,
#         test_metrics_check=False,
#     )
#     forecast_without_outlier = pd.read_csv(f"{tmpdirname}/results/forecast.csv")
#     input_vals_without_outlier = set(forecast_without_outlier["input_value"])
#     assert all(
#         item not in input_vals_without_outlier for item in outliers
#     ), "forecast file should not contain any outliers"

#     # switching off outlier_treatment
#     preprocessing_steps = {"missing_value_imputation": True, "outlier_treatment": False}
#     preprocessing = {"enabled": True, "steps": preprocessing_steps}
#     yaml_i["spec"]["preprocessing"] = preprocessing
#     run_yaml(
#         tmpdirname=tmpdirname,
#         yaml_i=yaml_i,
#         output_data_path=output_data_path,
#         test_metrics_check=False,
#     )
#     forecast_with_outlier = pd.read_csv(f"{tmpdirname}/results/forecast.csv")
#     input_vals_with_outlier = set(forecast_with_outlier["input_value"])
#     assert all(
#         item in input_vals_with_outlier for item in outliers
#     ), "forecast file should contain all the outliers"


@pytest.mark.parametrize("model", MODELS)
def test_2_series(operator_setup, model):
    # Test w and w/o add data
    tmpdirname = operator_setup

    def split_df(df):
        # Splits Store col into Store and Store_test
        idx_split = df.shape[0] // 2
        df_a = df[:idx_split]
        df_a["Store_test"] = "A"

        df_b = df[idx_split:]
        df_b = df_b.rename({"Store": "Store_test"}, axis=1).reset_index(drop=True)
        df_b["Store"] = "A"
        return pd.concat([df_a, df_b]).reset_index(drop=True)

    hist_data = split_df(HISTORICAL_DATA)
    add_data = split_df(ADDITIONAL_DATA)
    test_data = split_df(TEST_DATA)

    print(f"hist_data: {hist_data}\nadd_data:{add_data}\ntest_data:{test_data}\n")

    historical_data_path, additional_data_path, test_data_path = setup_artificial_data(
        tmpdirname, hist_data, add_data, test_data
    )
    preprocessing_steps = {"missing_value_imputation": True, "outlier_treatment": False}
    yaml_i, output_data_path = populate_yaml(
        tmpdirname=tmpdirname,
        model=model,
        historical_data_path=historical_data_path,
        additional_data_path=additional_data_path,
        test_data_path=test_data_path,
        preprocessing={"enabled": True, "steps": preprocessing_steps},
    )
    with pytest.raises(DataMismatchError):
        # 4 columns in historical data, but only 1 cat col specified
        run_yaml(
            tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path
        )
    yaml_i["spec"].pop("target_category_columns")
    with pytest.raises(DataMismatchError):
        # 4 columns in historical data, but only no cat col specified
        run_yaml(
            tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path
        )
    yaml_i["spec"]["target_category_columns"] = ["Store", "Store_test"]
    run_yaml(tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path)


@pytest.mark.xfail()
@pytest.mark.parametrize("model", MODELS)
def test_all_series_failure(model):
    """
    Every model is mocked to throw error. This test checks that errors.json has correct error message and that report is
    generated
    """
    tmpdirname = operator_setup
    historical_data_path, additional_data_path = setup_faulty_rossman()
    yaml_i, output_data_path = populate_yaml(
        tmpdirname=tmpdirname,
        historical_data_path=historical_data_path,
        additional_data_path=additional_data_path,
    )
    preprocessing_steps = {"missing_value_imputation": True, "outlier_treatment": False}
    yaml_i["spec"]["model"] = model
    yaml_i["spec"]["horizon"] = 10
    yaml_i["spec"]["preprocessing"] = {"enabled": True, "steps": preprocessing_steps}
    if yaml_i["spec"].get("additional_data") is not None and model != "autots":
        yaml_i["spec"]["generate_explanations"] = True
    else:
        yaml_i["spec"]["generate_explanations"] = False
    if model == "autots":
        yaml_i["spec"]["model_kwargs"] = {"model_list": "superfast"}
    if model == "automlx":
        yaml_i["spec"]["model_kwargs"] = {"time_budget": 1}

    module_to_patch = {
        "arima": "pmdarima.auto_arima",
        "autots": "autots.AutoTS",
        "automlx": "automlx.Pipeline",
        "prophet": "prophet.Prophet",
        "neuralprophet": "neuralprophet.NeuralProphet",
    }
    with patch(
        module_to_patch[model], side_effect=Exception("Custom exception message")
    ):
        run(yaml_i, backend="operator.local", debug=False)

        report_path = f"{output_data_path}/report.html"
        assert os.path.exists(report_path), f"Report file not found at {report_path}"

        error_path = f"{output_data_path}/errors.json"
        assert os.path.exists(error_path), f"Error file not found at {error_path}"

        # Additionally, you can read the content of the error.json and assert its content
        with open(error_path, "r") as error_file:
            error_content = json.load(error_file)
            assert (
                "Custom exception message" in error_content["1"]["error"]
            ), "Error message mismatch"
            assert (
                "Custom exception message" in error_content["13"]["error"]
            ), "Error message mismatch"

        if yaml_i["spec"]["generate_explanations"]:
            global_fn = f"{tmpdirname}/results/global_explanation.csv"
            assert os.path.exists(
                global_fn
            ), f"Global explanation file not found at {report_path}"

            local_fn = f"{tmpdirname}/results/local_explanation.csv"
            assert os.path.exists(
                local_fn
            ), f"Local explanation file not found at {report_path}"


@pytest.mark.parametrize("model", MODELS)
def test_arima_automlx_errors(operator_setup, model):
    tmpdirname = operator_setup
    historical_data_path, additional_data_path = setup_faulty_rossman()
    yaml_i, output_data_path = populate_yaml(
        tmpdirname=tmpdirname,
        historical_data_path=historical_data_path,
        additional_data_path=additional_data_path,
    )

    """
    Arima was failing for constant trend when there are constant columns and when there are boolean columns .
    We added label encoding for boolean and are dropping columns with constant value for arima with constant trend.
    This test checks that report, metrics, explanations are generated for this case.
    """

    """
    series 13 in this data has missing dates and automlx fails for this with DatetimeIndex error. This test checks that
    outputs get generated and that error is shown in errors.json


    explanations generation is failing when boolean columns are passed.
    TypeError: ufunc 'isfinite' not supported for the input types, and the inputs could not be safely coerced
     any supported types according to the casting rule ''safe''
    Added label encoding before passing data to explainer
    """
    preprocessing_steps = {"missing_value_imputation": True, "outlier_treatment": False}
    yaml_i["spec"]["horizon"] = 10
    yaml_i["spec"]["preprocessing"] = preprocessing_steps
    yaml_i["spec"]["generate_explanations"] = True
    yaml_i["spec"]["model"] = model
    if model == "autots":
        yaml_i["spec"]["model_kwargs"] = {"model_list": "superfast"}
    if model == "automlx":
        yaml_i["spec"]["model_kwargs"] = {"time_budget": 1}
        yaml_i["spec"]["explanations_accuracy_mode"] = "AUTOMLX"

    run_yaml(
        tmpdirname=tmpdirname,
        yaml_i=yaml_i,
        output_data_path=output_data_path,
        test_metrics_check=False,
    )

    report_path = f"{tmpdirname}/results/report.html"
    assert os.path.exists(report_path), f"Report file not found at {report_path}"

    forecast_path = f"{tmpdirname}/results/forecast.csv"
    assert os.path.exists(forecast_path), f"Forecast file not found at {report_path}"
    assert not pd.read_csv(forecast_path).empty

    error_path = f"{tmpdirname}/results/errors.json"
    if model == "arima":
        assert not os.path.exists(error_path), f"Error file not found at {error_path}"
    elif model == "automlx":
        assert os.path.exists(error_path), f"Error file not found at {error_path}"
        with open(error_path, "r") as error_file:
            error_content = json.load(error_file)
            assert (
                "Input data does not have a consistent (in terms of diff) DatetimeIndex."
                in error_content["13"]["model_fitting"]["error"]
            ), f"Error message mismatch: {error_content}"

    if model not in ["autots", "automlx"]:  # , "lgbforecast"
        if yaml_i["spec"].get("explanations_accuracy_mode") != "AUTOMLX":
            global_fn = f"{tmpdirname}/results/global_explanation.csv"
            assert os.path.exists(
                global_fn
            ), f"Global explanation file not found at {report_path}"
            assert not pd.read_csv(global_fn, index_col=0).empty

        local_fn = f"{tmpdirname}/results/local_explanation.csv"
        assert os.path.exists(
            local_fn
        ), f"Local explanation file not found at {report_path}"
        assert not pd.read_csv(local_fn).empty


def test_smape_error():
    result = smape([0, 0, 0, 0], [0, 0, 0, 0])
    assert result == 0


@pytest.mark.parametrize("model", ["prophet"])
def test_pandas_historical_input(operator_setup, model):
    from ads.opctl.operator.lowcode.forecast.__main__ import operate

    historical_data_path, additional_data_path, _ = setup_artificial_data(
        operator_setup
    )
    yaml_i, output_data_path = populate_yaml(
        tmpdirname=operator_setup,
        historical_data_path=historical_data_path,
        additional_data_path=additional_data_path,
    )
    yaml_i["spec"]["horizon"] = HORIZON
    yaml_i["spec"]["model"] = model
    df = pd.read_csv(historical_data_path)
    yaml_i["spec"]["historical_data"].pop("url")
    yaml_i["spec"]["historical_data"]["data"] = df
    yaml_i["spec"]["historical_data"]["format"] = "pandas"

    operator_config = ForecastOperatorConfig.from_dict(yaml_i)
    operate(operator_config)
    assert pd.read_csv(additional_data_path)["Date"].equals(
        pd.read_csv(f"{operator_setup}/results/forecast.csv")["Date"]
    )


@pytest.mark.parametrize("model", ["prophet"])
def test_pandas_additional_input(operator_setup, model):
    from ads.opctl.operator.lowcode.forecast.__main__ import operate

    tmpdirname = operator_setup
    historical_data_path, additional_data_path = setup_small_rossman()
    yaml_i, output_data_path = populate_yaml(
        tmpdirname=tmpdirname,
        historical_data_path=historical_data_path,
        additional_data_path=additional_data_path,
    )
    yaml_i["spec"]["horizon"] = 10
    yaml_i["spec"]["model"] = model
    df = pd.read_csv(historical_data_path)
    yaml_i["spec"]["historical_data"].pop("url")
    yaml_i["spec"]["historical_data"]["data"] = df
    yaml_i["spec"]["historical_data"]["format"] = "pandas"

    df_add = pd.read_csv(additional_data_path)
    yaml_i["spec"]["additional_data"].pop("url")
    yaml_i["spec"]["additional_data"]["data"] = df_add
    yaml_i["spec"]["additional_data"]["format"] = "pandas"

    operator_config = ForecastOperatorConfig.from_dict(yaml_i)
    results = operate(operator_config)
    assert pd.read_csv(additional_data_path)["Date"].equals(
        pd.read_csv(f"{tmpdirname}/results/forecast.csv")["Date"]
    )
    forecast = results.get_forecast()
    metrics = results.get_metrics()
    test_metrics = results.get_test_metrics()


@pytest.mark.parametrize("model", ["prophet"])
def test_date_format(operator_setup, model):
    tmpdirname = operator_setup
    historical_data_path, additional_data_path = setup_small_rossman()
    yaml_i, output_data_path = populate_yaml(
        tmpdirname=tmpdirname,
        historical_data_path=historical_data_path,
        additional_data_path=additional_data_path,
    )
    yaml_i["spec"]["horizon"] = 10
    yaml_i["spec"]["model"] = model
    if model == "autots":
        yaml_i["spec"]["model_kwargs"] = {"model_list": "superfast"}

    run_yaml(
        tmpdirname=tmpdirname,
        yaml_i=yaml_i,
        output_data_path=output_data_path,
        test_metrics_check=False,
    )
    assert pd.read_csv(additional_data_path)["Date"].equals(
        pd.read_csv(f"{tmpdirname}/results/forecast.csv")["Date"]
    )


@pytest.mark.parametrize("model", MODELS)
def test_what_if_analysis(operator_setup, model):
    os.environ["TEST_MODE"] = "True"
    if model == "auto-select":
        pytest.skip("Skipping what-if scenario for auto-select")
    tmpdirname = operator_setup
    historical_data_path, additional_data_path = setup_small_rossman()
    additional_test_path = f"{tmpdirname}/additional_data.csv"
    historical_test_path = f"{tmpdirname}/historical_data.csv"
    historical_data = pd.read_csv(historical_data_path, parse_dates=["Date"])
    historical_filtered = historical_data[historical_data["Date"] > "2013-03-01"]
    additional_data = pd.read_csv(additional_data_path, parse_dates=["Date"])
    add_filtered = additional_data[additional_data["Date"] > "2013-03-01"]
    add_filtered.to_csv(f"{additional_test_path}", index=False)
    historical_filtered.to_csv(f"{historical_test_path}", index=False)

    yaml_i, output_data_path = populate_yaml(
        tmpdirname=tmpdirname,
        historical_data_path=historical_test_path,
        additional_data_path=additional_test_path,
        output_data_path=f"{tmpdirname}/{model}/results",
    )
    yaml_i["spec"]["horizon"] = 10
    yaml_i["spec"]["model"] = model
    yaml_i["spec"]["what_if_analysis"] = {
        "model_name": f"model_{model}",
        "model_display_name": f"test_{model}",
        "project_id": "test_project_id",
        "compartment_id": "test_compartment_id",
    }

    run_yaml(
        tmpdirname=tmpdirname,
        yaml_i=yaml_i,
        output_data_path=output_data_path,
        test_metrics_check=False,
    )
    report_path = f"{output_data_path}/report.html"
    deployment_metadata = f"{output_data_path}/deployment_info.json"
    assert os.path.exists(report_path), f"Report file not found at {report_path}"
    assert os.path.exists(
        deployment_metadata
    ), f"Deployment info file not found at {deployment_metadata}"


def test_auto_select(operator_setup):
    DATASET_PREFIX = f"{os.path.dirname(os.path.abspath(__file__))}/../data/timeseries/"
    tmpdirname = operator_setup
    historical_test_path = f"{DATASET_PREFIX}/dataset6.csv"
    model = "auto-select"
    yaml_i, output_data_path = populate_yaml(
        tmpdirname=tmpdirname,
        historical_data_path=historical_test_path,
        output_data_path=f"{tmpdirname}/{model}/results",
    )
    yaml_i["spec"].pop("additional_data")
    yaml_i["spec"]["horizon"] = 2
    yaml_i["spec"]["datetime_column"]["format"] = "%d-%m-%Y"
    yaml_i["spec"]["model"] = model
    yaml_i["spec"]["model_kwargs"] = {"model_list": ["prophet", "arima"]}

    run_yaml(
        tmpdirname=tmpdirname,
        yaml_i=yaml_i,
        output_data_path=output_data_path,
        test_metrics_check=False,
    )
    report_path = f"{output_data_path}/report.html"
    assert os.path.exists(report_path), f"Report file not found at {report_path}"


@pytest.mark.parametrize("model", ["prophet"])
def test_report_title(operator_setup, model):
    yaml_i = TEMPLATE_YAML.copy()
    yaml_i["spec"]["horizon"] = 10
    yaml_i["spec"]["model"] = model
    yaml_i["spec"]["historical_data"] = {"format": "pandas"}
    yaml_i["spec"]["target_column"] = TARGET_COL.name
    yaml_i["spec"]["datetime_column"]["name"] = HISTORICAL_DATETIME_COL.name
    yaml_i["spec"]["report_title"] = "Skibidi ADS Skibidi"
    yaml_i["spec"]["output_directory"]["url"] = operator_setup

    df = pd.concat([HISTORICAL_DATETIME_COL[:15], TARGET_COL[:15]], axis=1)
    yaml_i["spec"]["historical_data"]["data"] = df
    operator_config = ForecastOperatorConfig.from_dict(yaml_i)
    results = operate(operator_config)
    title_found = False
    with open(os.path.join(operator_setup, "report.html")) as f:
        for line in f:
            if "Skibidi ADS Skibidi" in line:
                title_found = True
    assert title_found, "Report Title was not set"


@pytest.mark.parametrize("model", ["prophet"])
def test_prophet_floor_cap(operator_setup, model):
    yaml_i = TEMPLATE_YAML.copy()
    yaml_i["spec"]["horizon"] = 10
    yaml_i["spec"]["model"] = model
    yaml_i["spec"]["historical_data"] = {"format": "pandas"}
    yaml_i["spec"]["datetime_column"]["name"] = HISTORICAL_DATETIME_COL.name
    yaml_i["spec"]["output_directory"]["url"] = operator_setup
    yaml_i["spec"]["target_column"] = "target"
    yaml_i["spec"]["model_kwargs"] = {"min": 0, "max": 20}

    target_column = pd.Series(np.arange(20, -6, -2), name="target")
    df = pd.concat(
        [HISTORICAL_DATETIME_COL[: len(target_column)], target_column], axis=1
    )
    yaml_i["spec"]["historical_data"]["data"] = df
    operator_config = ForecastOperatorConfig.from_dict(yaml_i)
    results = operate(operator_config)
    assert np.all(
        results.get_forecast()["forecast_value"].dropna() > 0
    ), "`min` not obeyed in prophet"
    assert np.all(
        results.get_forecast()["fitted_value"].dropna() > 0
    ), "`min` not obeyed in prophet"

    target_column = pd.Series(np.arange(-6, 20, 2), name="target")
    df = pd.concat(
        [HISTORICAL_DATETIME_COL[: len(target_column)], target_column], axis=1
    )
    yaml_i["spec"]["historical_data"]["data"] = df
    operator_config = ForecastOperatorConfig.from_dict(yaml_i)
    results = operate(operator_config)
    assert np.all(
        results.get_forecast()["forecast_value"].dropna() < 20
    ), "`max` not obeyed in prophet"
    assert np.all(
        results.get_forecast()["fitted_value"].dropna() < 20
    ), "`max` not obeyed in prophet"


def _check_results_obj(results):
    assert not results.get_forecast().empty
    assert not results.get_metrics().empty
    assert not results.get_global_explanations().empty
    assert not results.get_local_explanations().empty


def _check_no_skippable_files(yaml_i, check_report=True):
    files = os.listdir(yaml_i["spec"]["output_directory"]["url"])

    if "errors.json" in files:
        with open(
            os.path.join(yaml_i["spec"]["output_directory"]["url"], "errors.json")
        ) as f:
            assert False, f"Failed due to errors.json being created: {f.read()}"
    if check_report:
        assert "report.html" in files, "Failed to generate report"

    assert (
        "forecast.csv" not in files
    ), "Generated forecast file, but `generate_forecast_file` was set False"
    assert (
        "metrics.csv" not in files
    ), "Generated metrics file, but `generate_metrics_file` was set False"
    assert (
        "local_explanations.csv" not in files
    ), "Generated metrics file, but `generate_explanation_files` was set False"
    assert (
        "global_explanations.csv" not in files
    ), "Generated metrics file, but `generate_explanation_files` was set False"


@pytest.mark.parametrize("model", ["prophet"])
def test_generate_files(operator_setup, model):
    yaml_i = TEMPLATE_YAML.copy()
    yaml_i["spec"]["horizon"] = 3
    yaml_i["spec"]["model"] = model
    yaml_i["spec"]["historical_data"] = {"format": "pandas"}
    yaml_i["spec"]["additional_data"] = {"format": "pandas"}
    yaml_i["spec"]["target_column"] = TARGET_COL.name
    yaml_i["spec"]["datetime_column"]["name"] = HISTORICAL_DATETIME_COL.name
    yaml_i["spec"]["output_directory"]["url"] = operator_setup
    yaml_i["spec"]["generate_explanation_files"] = False
    yaml_i["spec"]["generate_forecast_file"] = False
    yaml_i["spec"]["generate_metrics_file"] = False
    yaml_i["spec"]["generate_explanations"] = True
    yaml_i["spec"]["model_kwargs"] = {"min": 0, "max": 20}

    df = pd.concat([HISTORICAL_DATETIME_COL[:15], TARGET_COL[:15]], axis=1)
    df_add = pd.concat([HISTORICAL_DATETIME_COL[:18], ADD_COLS[:18]], axis=1)
    yaml_i["spec"]["historical_data"]["data"] = df
    yaml_i["spec"]["additional_data"]["data"] = df_add
    operator_config = ForecastOperatorConfig.from_dict(yaml_i)
    results = operate(operator_config)
    _check_results_obj(results)
    _check_no_skippable_files(yaml_i)

    yaml_i["spec"].pop("generate_explanation_files")
    yaml_i["spec"].pop("generate_forecast_file")
    yaml_i["spec"].pop("generate_metrics_file")
    operator_config = ForecastOperatorConfig.from_dict(yaml_i)
    results = operate(operator_config)
    _check_results_obj(results)
    files = os.listdir(yaml_i["spec"]["output_directory"]["url"])
    if "errors.json" in files:
        with open(
            os.path.join(yaml_i["spec"]["output_directory"]["url"], "errors.json")
        ) as f:
            print(f"Errors in build! {f.read()}")
            assert False, "Failed due to errors.json being created"
    assert "report.html" in files, "Failed to generate report"
    assert "forecast.csv" in files, "Failed to generate forecast file"
    assert "metrics.csv" in files, "Failed to generated metrics file"
    assert "local_explanation.csv" in files, "Failed to generated local expl file"
    assert "global_explanation.csv" in files, "Failed to generated global expl file"

    # Test that the results object still generates when report.html has an error
    yaml_i["spec"]["output_directory"]["url"] = "s3://test@test/test_dir"
    operator_config = ForecastOperatorConfig.from_dict(yaml_i)
    results = operate(operator_config)
    _check_results_obj(results)


if __name__ == "__main__":
    pass
