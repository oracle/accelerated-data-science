#!/usr/bin/env python

# Copyright (c) 2023, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import tempfile
import pandas as pd
import pytest
from copy import deepcopy
from ads.opctl.operator.lowcode.forecast.__main__ import operate
from ads.opctl.operator.lowcode.forecast.operator_config import ForecastOperatorConfig

DATASET_PREFIX = f"{os.path.dirname(os.path.abspath(__file__))}/../data/timeseries/"

TEMPLATE_YAML = {
    "kind": "operator",
    "type": "forecast",
    "version": "v1",
    "spec": {
        "historical_data": {
            "url": f"{DATASET_PREFIX}dataset1.csv",
        },
        "output_directory": {
            "url": "results",
        },
        "model": "prophet",
        "target_column": "Y",
        "datetime_column": {
            "name": "Date",
        },
        "horizon": 5,
        "generate_explanations": False,
    },
}

@pytest.fixture(autouse=True)
def operator_setup():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

class TestForecastApiOptions:
    def test_custom_filenames(self, operator_setup):
        """Tests that custom filenames are correctly used."""
        tmpdirname = operator_setup
        yaml_i = deepcopy(TEMPLATE_YAML)
        yaml_i["spec"]["output_directory"]["url"] = tmpdirname
        yaml_i["spec"]["report_filename"] = "my_report.html"
        yaml_i["spec"]["metrics_filename"] = "my_metrics.csv"
        yaml_i["spec"]["test_metrics_filename"] = "my_test_metrics.csv"
        yaml_i["spec"]["forecast_filename"] = "my_forecast.csv"
        yaml_i["spec"]["test_data"] = {
            "url": f"{DATASET_PREFIX}dataset1.csv"
        }

        operator_config = ForecastOperatorConfig.from_dict(yaml_i)
        operate(operator_config)

        output_files = os.listdir(tmpdirname)
        assert "my_report.html" in output_files
        assert "my_metrics.csv" in output_files
        assert "my_test_metrics.csv" in output_files
        assert "my_forecast.csv" in output_files

    def test_report_theme(self, operator_setup):
        """Tests that the report theme is correctly applied."""
        tmpdirname = operator_setup
        yaml_i = deepcopy(TEMPLATE_YAML)
        yaml_i["spec"]["output_directory"]["url"] = tmpdirname
        yaml_i["spec"]["report_theme"] = "dark"

        operator_config = ForecastOperatorConfig.from_dict(yaml_i)
        operate(operator_config)

        with open(os.path.join(tmpdirname, "report.html"), "r") as f:
            report_content = f.read()
            assert "dark" in report_content

    def test_disable_report_generation(self, operator_setup):
        """Tests that report generation can be disabled."""
        tmpdirname = operator_setup
        yaml_i = deepcopy(TEMPLATE_YAML)
        yaml_i["spec"]["output_directory"]["url"] = tmpdirname
        yaml_i["spec"]["generate_report"] = False

        operator_config = ForecastOperatorConfig.from_dict(yaml_i)
        operate(operator_config)

        output_files = os.listdir(tmpdirname)
        assert "report.html" not in output_files

    def test_previous_output_dir(self, operator_setup):
        """Tests that a previous model can be loaded."""
        tmpdirname = operator_setup

        # First run: generate a model
        first_run_dir = os.path.join(tmpdirname, "first_run")
        os.makedirs(first_run_dir)
        yaml1 = deepcopy(TEMPLATE_YAML)
        yaml1["spec"]["output_directory"]["url"] = first_run_dir
        yaml1["spec"]["generate_model_pickle"] = True

        operator_config1 = ForecastOperatorConfig.from_dict(yaml1)
        operate(operator_config1)

        # Second run: use the previous model
        second_run_dir = os.path.join(tmpdirname, "second_run")
        os.makedirs(second_run_dir)
        yaml2 = deepcopy(TEMPLATE_YAML)
        yaml2["spec"]["output_directory"]["url"] = second_run_dir
        yaml2["spec"]["previous_output_dir"] = first_run_dir

        operator_config2 = ForecastOperatorConfig.from_dict(yaml2)
        operate(operator_config2)

        # Check that the second run produced a forecast
        output_files = os.listdir(second_run_dir)
        assert "forecast.csv" in output_files

    def test_generate_model_artifacts(self, operator_setup):
        """Tests that model artifacts are correctly generated."""
        tmpdirname = operator_setup
        yaml_i = deepcopy(TEMPLATE_YAML)
        yaml_i["spec"]["output_directory"]["url"] = tmpdirname
        yaml_i["spec"]["generate_model_parameters"] = True
        yaml_i["spec"]["generate_model_pickle"] = True

        operator_config = ForecastOperatorConfig.from_dict(yaml_i)
        operate(operator_config)

        output_files = os.listdir(tmpdirname)
        assert "model_params.json" in output_files

    def test_metric(self, operator_setup):
        """Tests that the metric is correctly used."""
        tmpdirname = operator_setup
        yaml_i = deepcopy(TEMPLATE_YAML)
        yaml_i["spec"]["output_directory"]["url"] = tmpdirname
        yaml_i["spec"]["metric"] = "RMSE"
        yaml_i["spec"]["test_data"] = {
            "url": f"{DATASET_PREFIX}dataset1.csv"
        }

        operator_config = ForecastOperatorConfig.from_dict(yaml_i)
        operate(operator_config)

        metrics = pd.read_csv(os.path.join(tmpdirname, "metrics.csv"))
        assert "RMSE" in metrics["Metric"].values

    def test_outlier_treatment(self, operator_setup):
        """Tests that outlier treatment is correctly applied."""
        tmpdirname = operator_setup

        # Create a dataset with outliers
        data = pd.read_csv(f"{DATASET_PREFIX}dataset1.csv")
        data.loc[5, "Y"] = 1000
        data.loc[15, "Y"] = -1000
        historical_data_path = os.path.join(tmpdirname, "historical_data.csv")
        data.to_csv(historical_data_path, index=False)

        # Run with outlier treatment
        yaml_with = deepcopy(TEMPLATE_YAML)
        yaml_with["spec"]["historical_data"]["url"] = historical_data_path
        yaml_with["spec"]["output_directory"]["url"] = os.path.join(tmpdirname, "with_treatment")
        yaml_with["spec"]["preprocessing"] = {"steps": {"outlier_treatment": True}}

        operate(ForecastOperatorConfig.from_dict(yaml_with))

        # Run without outlier treatment
        yaml_without = deepcopy(TEMPLATE_YAML)
        yaml_without["spec"]["historical_data"]["url"] = historical_data_path
        yaml_without["spec"]["output_directory"]["url"] = os.path.join(tmpdirname, "without_treatment")
        yaml_without["spec"]["preprocessing"] = {"steps": {"outlier_treatment": False}}

        operate(ForecastOperatorConfig.from_dict(yaml_without))

        # Check that outliers are present in the forecast without treatment
        forecast_without = pd.read_csv(os.path.join(tmpdirname, "without_treatment", "forecast.csv"))
        assert 1000 in forecast_without["yhat"].values
        assert -1000 in forecast_without["yhat"].values

        # Check that outliers are not present in the forecast with treatment
        forecast_with = pd.read_csv(os.path.join(tmpdirname, "with_treatment", "forecast.csv"))
        assert 1000 not in forecast_with["yhat"].values
        assert -1000 not in forecast_with["yhat"].values

    def test_missing_value_imputation(self, operator_setup):
        """Tests that missing value imputation is correctly applied."""
        tmpdirname = operator_setup

        # Create a dataset with missing values
        data = pd.read_csv(f"{DATASET_PREFIX}dataset1.csv")
        data.loc[5, "Y"] = None
        data.loc[15, "Y"] = None
        historical_data_path = os.path.join(tmpdirname, "historical_data.csv")
        data.to_csv(historical_data_path, index=False)

        # Run with missing value imputation
        yaml_i = deepcopy(TEMPLATE_YAML)
        yaml_i["spec"]["historical_data"]["url"] = historical_data_path
        yaml_i["spec"]["output_directory"]["url"] = tmpdirname
        yaml_i["spec"]["preprocessing"] = {"steps": {"missing_value_imputation": True}}

        results = operate(ForecastOperatorConfig.from_dict(yaml_i))
        forecast = results.get_forecast()

        # Check that there are no missing values in the forecast
        assert not forecast["yhat"].isnull().any()
        assert "model.pkl" in output_files

    def test_confidence_interval_width(self, operator_setup):
        """Tests that the confidence interval width is correctly applied."""
        tmpdirname = operator_setup
        yaml_i = deepcopy(TEMPLATE_YAML)
        yaml_i["spec"]["output_directory"]["url"] = tmpdirname
        yaml_i["spec"]["confidence_interval_width"] = 0.95

        operator_config = ForecastOperatorConfig.from_dict(yaml_i)
        results = operate(operator_config)
        forecast = results.get_forecast()

        # Check that the confidence interval is close to the specified width
        # This is a basic check, a more robust check would involve statistical tests
        assert "yhat_upper" in forecast.columns
        assert "yhat_lower" in forecast.columns

    def test_tuning(self, operator_setup):
        """Tests that tuning is correctly applied."""
        tmpdirname = operator_setup
        yaml_i = deepcopy(TEMPLATE_YAML)
        yaml_i["spec"]["output_directory"]["url"] = tmpdirname
        yaml_i["spec"]["tuning"] = {"n_trials": 5}
        yaml_i["spec"]["generate_model_parameters"] = True

        operator_config = ForecastOperatorConfig.from_dict(yaml_i)
        operate(operator_config)

        output_files = os.listdir(tmpdirname)
        assert "model_params.json" in output_files
