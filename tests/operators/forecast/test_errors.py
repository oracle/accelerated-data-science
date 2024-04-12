#!/usr/bin/env python
from unittest.mock import patch

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import yaml
import tempfile
import subprocess
import pandas as pd
import numpy as np
import pytest
from darts import datasets as d_datasets
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

from ads.opctl.operator.lowcode.forecast.utils import smape
from ads.opctl.operator.cmd import run
import os
import json

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


def run_yaml(tmpdirname, yaml_i, output_data_path, test_metrics_check=True):
    run(yaml_i, backend="operator.local", debug=True)
    subprocess.run(f"ls -a {output_data_path}", shell=True)

    if test_metrics_check:
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


def setup_faulty_rossman():
    curr_dir = pathlib.Path(__file__).parent.resolve()
    data_folder = f"{curr_dir}/../data/"
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
    yaml_i, output_data_path = populate_yaml(
        tmpdirname=tmpdirname,
        model=model,
        historical_data_path=historical_data_path,
        additional_data_path=additional_data_path,
        test_data_path=test_data_path,
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

    yaml_i["spec"]["model"] = model
    yaml_i['spec']['horizon'] = 10
    yaml_i['spec']['preprocessing'] = True
    if yaml_i["spec"].get("additional_data") is not None and model != "autots":
        yaml_i["spec"]["generate_explanations"] = True
    if model == "autots":
        yaml_i["spec"]["model_kwargs"] = {"model_list": "superfast"}
    if model == "automlx":
        yaml_i["spec"]["model_kwargs"] = {"time_budget": 1}

    module_to_patch = {
        "arima": 'pmdarima.auto_arima',
        "autots": 'autots.AutoTS',
        "automlx": 'automlx.Pipeline',
        "prophet": 'prophet.Prophet',
        "neuralprophet": 'neuralprophet.NeuralProphet'
    }
    with patch(module_to_patch[model], side_effect=Exception("Custom exception message")):

        run(yaml_i, backend="operator.local", debug=False)

        report_path = f"{output_data_path}/report.html"
        assert os.path.exists(report_path), f"Report file not found at {report_path}"

        error_path = f"{output_data_path}/errors.json"
        assert os.path.exists(error_path), f"Error file not found at {error_path}"

        # Additionally, you can read the content of the error.json and assert its content
        with open(error_path, 'r') as error_file:
            error_content = json.load(error_file)
            assert "Custom exception message" in error_content["1"]["error"], "Error message mismatch"
            assert "Custom exception message" in error_content["13"]["error"], "Error message mismatch"

        if yaml_i["spec"]["generate_explanations"]:
            global_fn = f"{tmpdirname}/results/global_explanation.csv"
            assert os.path.exists(global_fn), f"Global explanation file not found at {report_path}"

            local_fn = f"{tmpdirname}/results/local_explanation.csv"
            assert os.path.exists(local_fn), f"Local explanation file not found at {report_path}"

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
    """

    """
    explanations generation is failing when boolean columns are passed. So we added label_encode before passing data to
     explainer
    """

    yaml_i['spec']['horizon'] = 10
    yaml_i['spec']['preprocessing'] = True
    yaml_i['spec']['generate_explanations'] = True
    yaml_i['spec']['model'] = model

    run_yaml(tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path, test_metrics_check=False)

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
        with open(error_path, 'r') as error_file:
            error_content = json.load(error_file)
            assert "Input data does not have a consistent (in terms of diff) DatetimeIndex." in error_content["13"][
                "error"], "Error message mismatch"

    if model != "autots":
        global_fn = f"{tmpdirname}/results/global_explanation.csv"
        assert os.path.exists(global_fn), f"Global explanation file not found at {report_path}"

        local_fn = f"{tmpdirname}/results/local_explanation.csv"
        assert os.path.exists(local_fn), f"Local explanation file not found at {report_path}"

        glb_expl = pd.read_csv(global_fn, index_col=0)
        loc_expl = pd.read_csv(local_fn)
        assert not glb_expl.empty
        assert not loc_expl.empty


def test_smape_error():
    result = smape([0, 0, 0, 0], [0, 0, 0, 0])
    assert result == 0


@pytest.mark.parametrize("model", MODELS)
def test_date_format(operator_setup, model):
    tmpdirname = operator_setup
    historical_data_path, additional_data_path = setup_small_rossman()
    yaml_i, output_data_path = populate_yaml(
        tmpdirname=tmpdirname,
        historical_data_path=historical_data_path,
        additional_data_path=additional_data_path,
    )
    yaml_i['spec']['horizon'] = 10
    yaml_i["spec"]["model"] = model
    if model == "autots":
        yaml_i["spec"]["model_kwargs"] = {"model_list": "superfast"}

    run_yaml(tmpdirname=tmpdirname, yaml_i=yaml_i, output_data_path=output_data_path, test_metrics_check=False)
    assert pd.read_csv(additional_data_path)['Date'].equals(pd.read_csv(f"{tmpdirname}/results/forecast.csv")['Date'])


if __name__ == "__main__":
    pass
