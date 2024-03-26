#!/usr/bin/env python

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
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
from ads.opctl.operator.cmd import run
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


@pytest.fixture(autouse=True)
def operator_setup():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


def run_yaml(tmpdirname, yaml_i, output_data_path):
    run(yaml_i, backend="operator.local", debug=True)
    subprocess.run(f"ls -a {output_data_path}", shell=True)

    if 'test_data' in yaml_i['spec']:
        test_metrics = pd.read_csv(f"{tmpdirname}/results/test_metrics.csv")
        print(test_metrics)
    train_metrics = pd.read_csv(f"{tmpdirname}/results/metrics.csv")
    print(train_metrics)


def populate_yaml(
    tmpdirname=None,
    model="auto",
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
        yaml_i["spec"]["model_kwargs"] = {"time_budget": 50}

    return yaml_i, output_data_path


def run_operator(
    tmpdirname=None,
    model="auto",
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


@pytest.mark.parametrize("model", MODELS)
def test_rossman(operator_setup, model):
    run_operator(
        tmpdirname=operator_setup,
        model=model,
    )


@pytest.mark.parametrize("model", MODELS)
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
        preprocessing={"enabled": False}
    )
    with pytest.raises(DataMismatchError):
        run_yaml(
            tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path
        )
    yaml_i["spec"].pop("target_category_columns")
    run_yaml(tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path)
    add_data = yaml_i["spec"].pop("additional_data")
    run_yaml(tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path)
    test_data = yaml_i["spec"].pop("test_data")
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


def test_disabling_outlier_treatment(operator_setup):
    tmpdirname = operator_setup
    NUM_ROWS = 100
    hist_data_0 = pd.concat(
        [
            HISTORICAL_DATETIME_COL[: NUM_ROWS - HORIZON],
            TARGET_COL[: NUM_ROWS - HORIZON],
        ],
        axis=1,
    )
    outliers = [1000, -800]
    hist_data_0.at[40, 'Sales'] = outliers[0]
    hist_data_0.at[75, 'Sales'] = outliers[1]
    historical_data_path, additional_data_path, test_data_path = setup_artificial_data(
        tmpdirname, hist_data_0
    )

    yaml_i, output_data_path = populate_yaml(
        tmpdirname=tmpdirname,
        model="arima",
        historical_data_path=historical_data_path
    )
    yaml_i["spec"].pop("target_category_columns")
    yaml_i["spec"].pop("additional_data")

    # running default pipeline where outlier will be treated
    run_yaml(tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path)
    forecast_without_outlier = pd.read_csv(f"{tmpdirname}/results/forecast.csv")
    input_vals_without_outlier = set(forecast_without_outlier['input_value'])
    assert all(
        item not in input_vals_without_outlier for item in outliers), "forecast file should not contain any outliers"

    # switching off outlier_treatment
    preprocessing_steps = {"missing_value_imputation": True, "outlier_treatment": False}
    preprocessing = {"enabled": True, "steps": preprocessing_steps}
    yaml_i["spec"]["preprocessing"] = preprocessing
    run_yaml(tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path)
    forecast_with_outlier = pd.read_csv(f"{tmpdirname}/results/forecast.csv")
    input_vals_with_outlier = set(forecast_with_outlier['input_value'])
    assert all(
        item in input_vals_with_outlier for item in outliers), "forecast file should contain all the outliers"


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
        preprocessing={"enabled": True, "steps": preprocessing_steps}
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


if __name__ == "__main__":
    pass
