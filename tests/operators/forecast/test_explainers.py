#!/usr/bin/env python

# Copyright (c) 2023, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import os
import tempfile
from copy import deepcopy

import numpy as np
import pytest

from ads.opctl.operator.lowcode.forecast.__main__ import operate as forecast_operate
from ads.opctl.operator.lowcode.forecast.operator_config import (
    ForecastOperatorConfig,
)
from tests.operators.common.timeseries_syn_gen import TimeSeriesGenerator

MODELS = [
    "arima",
    # "automlx", # FIXME: automlx is failing, no errors
    "prophet",
    "neuralprophet",
    "theta",
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
        "generate_explanations": True,
    },
}


def generate_datasets(freq, horizon, num_series, num_points=100, seed=42):
    """
    Generate datasets using TimeSeriesGenerator.

    Parameters:
    - freq: Frequency of the datetime column.
    - horizon: Forecast horizon.
    - num_series: Number of different time series to generate.
    - num_points: Number of data points per time series.
    - seed: Random seed for deterministic data generation.

    Returns:
    - Tuple of two DataFrames: primary and additional.
    - Tuple of two lists: column names of primary and additional DataFrames.
    """
    generator = TimeSeriesGenerator(
        num_series=num_series,
        num_points=num_points,
        freq=freq,
        horizon=horizon,
        seed=seed,
    )
    primary, additional = generator.generate_timeseries_data()
    primary_columns = primary.columns.tolist()
    additional_columns = additional.columns.tolist()
    return primary, additional, primary_columns, additional_columns


def test_generate_datasets():
    primary, additional, primary_columns, additional_columns = generate_datasets(
        freq="D", horizon=5, num_series=10
    )
    assert len(primary) == 10 * 100
    assert len(additional) == 10 * (100 + 5)
    assert "target" in primary_columns
    assert "target" not in additional_columns


def setup_test_data(
    model, freq, num_series, horizon=5, num_points=100, seed=42, include_additional=True
):
    """
    Setup test data for the given parameters.

    Parameters:
    - model: The forecasting model to use.
    - freq: Frequency of the datetime column.
    - num_series: Number of different time series to generate.
    - horizon: Forecast horizon.
    - num_points: Number of data points per time series.
    - seed: Random seed for deterministic data generation.
    - include_additional: Boolean flag to include additional data.

    Returns:
    - Tuple containing primary, additional datasets and the operator configuration.
    """
    primary, additional, _, _ = generate_datasets(
        freq=freq,
        horizon=horizon,
        num_series=num_series,
        num_points=num_points,
        seed=seed,
    )

    yaml_i = deepcopy(TEMPLATE_YAML)
    yaml_i["spec"]["historical_data"].pop("url")
    yaml_i["spec"]["historical_data"]["data"] = primary
    yaml_i["spec"]["historical_data"]["format"] = "pandas"

    if include_additional:
        yaml_i["spec"]["additional_data"] = {"data": additional, "format": "pandas"}

    yaml_i["spec"]["model"] = model
    yaml_i["spec"]["target_column"] = "target"
    yaml_i["spec"]["datetime_column"]["name"] = "ds"
    yaml_i["spec"]["target_category_columns"] = ["series_id"]
    yaml_i["spec"]["horizon"] = horizon

    if model == "automlx":
        yaml_i["spec"]["explanations_accuracy_mode"] = "AUTOMLX"

    operator_config = ForecastOperatorConfig.from_dict(yaml_i)
    return primary, additional, operator_config


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("freq", ["D", "W", "M", "H", "T"])
@pytest.mark.parametrize("num_series", [1, 3])
def test_explanations_output_and_columns(model, freq, num_series):
    """
    Test the global and local explanations for different models, frequencies, and number of series.
    Also test that the explanation output contains all the columns from the additional dataset.

    Parameters:
    - model: The forecasting model to use.
    - freq: Frequency of the datetime column.
    - num_series: Number of different time series to generate.
    """
    if model == "automlx" and freq == "T":
        pytest.skip(
            "Skipping 'T' frequency for 'automlx' model. automlx requires data with a frequency of at least one hour"
        )
    if model == "neuralprophet":
        pytest.skip("Skipping 'neuralprophet' model as it takes a long time to finish")

    _, additional, operator_config = setup_test_data(model, freq, num_series)

    results = forecast_operate(operator_config)

    global_explanations = results.get_global_explanations()
    local_explanations = results.get_local_explanations()

    assert not (global_explanations.isna()).all().all(), (
        "Global explanations contain NaN values"
    )
    assert not (global_explanations == 0).all().all(), (
        "Global explanations contain only 0 values"
    )
    assert not (local_explanations.isna()).all().all(), (
        "Local explanations contain NaN values"
    )
    assert not (local_explanations == 0).all().all(), (
        "Local explanations contain only 0 values"
    )

    additional_columns = list(
        set(additional.columns.tolist())
        - set(operator_config.spec.target_category_columns)
        - {operator_config.spec.datetime_column.name}
    )
    for column in additional_columns:
        assert column in global_explanations.T.columns, (
            f"Column {column} missing in global explanations"
        )
        assert column in local_explanations.columns, (
            f"Column {column} missing in local explanations"
        )


@pytest.mark.parametrize("model", MODELS)  # MODELS
@pytest.mark.parametrize("num_series", [1])
def test_explanations_filenames(model, num_series):
    """
    Test that the global and local explanation filenames can be changed and are as specified.

    Parameters:
    - model: The forecasting model to use.
    - num_series: Number of different time series to generate.
    """
    if model == "neuralprophet":
        pytest.skip("Skipping 'neuralprophet' model as it takes a long time to finish")

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_directory = tmpdirname
        global_explanation_filename = "custom_global_explanation.csv"
        local_explanation_filename = "custom_local_explanation.csv"

        _, additional, operator_config = setup_test_data(model, "D", num_series)
        operator_config.spec.output_directory.url = output_directory
        operator_config.spec.global_explanation_filename = global_explanation_filename
        operator_config.spec.local_explanation_filename = local_explanation_filename

        results = forecast_operate(operator_config)
        assert not results.get_global_explanations().empty, (
            "Error generating Global Expl"
        )
        assert not results.get_local_explanations().empty, "Error generating Local Expl"

        if model == "auto-select-series":
            # List all files in output directory
            files = os.listdir(output_directory)
            # Find all explanation files
            global_explanation_files = [
                f
                for f in files
                if f.startswith("custom_global_explanation_") and f.endswith(".csv")
            ]
            local_explanation_files = [
                f
                for f in files
                if f.startswith("custom_local_explanation_") and f.endswith(".csv")
            ]

            # Should have at least one file of each type
            assert len(global_explanation_files) > 0, (
                "No global explanation files found for auto-select-series"
            )
            assert len(local_explanation_files) > 0, (
                "No local explanation files found for auto-select-series"
            )

            # Check each file exists
            for gfile in global_explanation_files:
                gpath = os.path.join(output_directory, gfile)
                assert os.path.exists(gpath), (
                    f"Global explanation file not found at {gpath}"
                )

            for lfile in local_explanation_files:
                lpath = os.path.join(output_directory, lfile)
                assert os.path.exists(lpath), (
                    f"Local explanation file not found at {lpath}"
                )
        else:
            global_explanation_path = os.path.join(
                output_directory, global_explanation_filename
            )
            local_explanation_path = os.path.join(
                output_directory, local_explanation_filename
            )

            assert os.path.exists(global_explanation_path), (
                f"Global explanation file not found at {global_explanation_path}"
            )
            assert os.path.exists(local_explanation_path), (
                f"Local explanation file not found at {local_explanation_path}"
            )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("num_series", [1])
def test_explanations_no_additional_data(model, num_series, caplog):
    """
    Test that the explanations fail when no additional dataset is provided.

    Parameters:
    - model: The forecasting model to use.
    - num_series: Number of different time series to generate.
    """
    if model == "neuralprophet":
        pytest.skip("Skipping 'neuralprophet' model as it takes a long time to finish")

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_directory = tmpdirname

        _, _, operator_config = setup_test_data(
            model, "D", num_series, include_additional=False
        )
        operator_config.spec.output_directory.url = output_directory

        forecast_operate(operator_config)

        assert any(
            "Unable to generate explanations as there is no additional data passed in. Either set generate_explanations to False, or pass in additional data."
            in message
            for message in caplog.messages
        ), "Required warning message not found in logs"


MODES = ["BALANCED", "HIGH_ACCURACY"]


@pytest.mark.skip(reason="Disabled by default. Enable to run this test.")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("num_series", [3])
def test_explanations_accuracy_mode(mode, model, num_series):
    """
    Test the explanations_accuracy_mode for different models and modes.

    Parameters:
    - mode: The explanations accuracy mode to use.
    - model: The forecasting model to use.
    - num_series: Number of different time series to generate.
    """
    if model == "neuralprophet":
        pytest.skip("Skipping 'neuralprophet' model as it takes a long time to finish")

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_directory = tmpdirname

        _, _, operator_config = setup_test_data(model, "D", num_series)
        operator_config.spec.output_directory.url = output_directory
        operator_config.spec.explanations_accuracy_mode = mode

        forecast_operate(operator_config)

        global_explanation_path = os.path.join(
            output_directory, operator_config.spec.global_explanation_filename
        )
        local_explanation_path = os.path.join(
            output_directory, operator_config.spec.local_explanation_filename
        )

        assert os.path.exists(global_explanation_path), (
            f"Global explanation file not found at {global_explanation_path}"
        )
        assert os.path.exists(local_explanation_path), (
            f"Local explanation file not found at {local_explanation_path}"
        )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("num_series", [1])
@pytest.mark.parametrize("freq", ["D"])
def test_explanations_values(model, num_series, freq):
    """
    Test that the sum of local explanations is near the actual forecasted values.

    Parameters:
    - model: The forecasting model to use.
    - num_series: Number of different time series to generate.
    - freq: Frequency of the datetime column.
    """
    if model == "neuralprophet":
        pytest.skip("Skipping 'neuralprophet' model as it takes a long time to finish")

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_directory = tmpdirname

        _, _, operator_config = setup_test_data(model, freq, num_series)
        operator_config.spec.output_directory.url = output_directory

        results = forecast_operate(operator_config)

        local_explanations = results.get_local_explanations()
        forecast = results.get_forecast()

        if model == "automlx":
            pytest.xfail("automlx model does not provide fitted values")

        # Check decimal precision for local explanations
        local_numeric = local_explanations.select_dtypes(include=["int64", "float64"])
        assert np.allclose(local_numeric, np.round(local_numeric, 4), atol=1e-8), (
            "Local explanations have values with more than 4 decimal places"
        )

        # Check decimal precision for global explanations
        global_explanations = results.get_global_explanations()
        global_numeric = global_explanations.select_dtypes(include=["int64", "float64"])
        assert np.allclose(global_numeric, np.round(global_numeric, 4), atol=1e-8), (
            "Global explanations have values with more than 4 decimal places"
        )

        local_explain_vals = local_numeric.sum(axis=1) + forecast.fitted_value.mean()
        assert np.allclose(
            local_explain_vals,
            forecast[-operator_config.spec.horizon :]["forecast_value"],
            rtol=0.1
            * np.max(
                np.abs(forecast[-operator_config.spec.horizon :]["forecast_value"])
            ),
        ), "Sum of local explanations is not close to the forecasted values"
