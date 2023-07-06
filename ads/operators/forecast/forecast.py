#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import json
import logging
import os
import sys
import datapane as dp
from prophet.plot import add_changepoints_to_plot
import pandas as pd
from urllib.parse import urlparse
import json
import yaml
import time
import ads
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_percentage_error,
    explained_variance_score,
    r2_score,
    mean_squared_error,
)
from sklearn.datasets import load_files
import oci
import time
from datetime import datetime
import fsspec

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

try:
    from ads.operators.forecast.prophet import operate as prophet_operate
    from ads.operators.forecast.prophet import get_prophet_report
    from ads.operators.forecast.neural_prophet import operate as neuralprophet_operate
    from ads.operators.forecast.neural_prophet import get_neuralprophet_report
    from ads.operators.forecast.arima import operate as arima_operate
    from ads.operators.forecast.arima import get_arima_report
    from ads.operators.forecast.utils import (
        evaluate_metrics,
        test_evaluate_metrics,
        get_forecast_plots,
    )
except Exception as ex:
    print(
        "Please run `pip install oracle-ads[forecast]` to install "
        "the required dependencies for ADS CLI."
    )
    logger.debug(ex)
    logger.debug(traceback.format_exc())
    exit()

AVAILABLE_MODELS = ["prophet", "neuralprophet", "arima"]


class ForecastOperator:
    def __init__(self, args):
        self.args = args
        assert args["kind"] == "operator"
        assert args["type"] == "forecast"
        assert args["version"] == 1
        self.historical_data = args["spec"]["historical_data"]
        self.additional_data = args["spec"].get("additional_data", dict())
        self.output_directory = args["spec"]["output_directory"]
        self.model = args["spec"]["model"].lower()
        self.target_columns = (
            None  # This will become [target__category1__category2 ...]
        )
        self.target_column = args["spec"]["target_column"]
        self.original_target_column = args["spec"]["target_column"]
        self.target_category_columns = args["spec"]["target_category_columns"]
        self.test_data = args["spec"]["test_data"]
        self.datetime_column = args["spec"]["datetime_column"]
        self.horizon = args["spec"]["horizon"]
        self.report_file_name = args["spec"].get(
            "report_file_name",
            os.path.join(self.output_directory["url"], "report.html"),
        )
        self.selected_metric = args["spec"].get("metric", "smape").lower()
        if args["spec"].get("tuning") is not None:
            self.perform_tuning = True
            self.num_tuning_trials = int(args["spec"]["tuning"].get("n_trials", 10))
        else:
            self.perform_tuning = False
            self.num_tuning_trials = 1

        # TODO: clean up
        self.input_filename = self.historical_data["url"]
        self.additional_filename = self.additional_data.get("url")
        self.output_filename = os.path.join(
            self.output_directory["url"], "forecast.csv"
        )
        self.test_filename = self.test_data["url"]
        self.ds_column = self.datetime_column.get("name")
        self.datetime_format = self.datetime_column.get("format")
        if args["execution"]["auth"] == "api_key":
            self.storage_options = {
                "profile": self.args["execution"].get("oci_profile"),
                "config": self.args["execution"].get("oci_config"),
            }
        else:
            # TODO: should we differ to ads config
            self.storage_options = dict()
        output_dir_name = self.output_directory["url"]
        output_dir_protocol = fsspec.utils.get_protocol(output_dir_name)

        if output_dir_protocol == "file":
            from pathlib import Path

            Path(output_dir_name).mkdir(parents=True, exist_ok=True)
        else:
            try:
                fs = fsspec.filesystem(output_dir_protocol, **self.storage_options)
                fs.mkdir(output_dir_name)
            except:
                logger.debug(
                    f"Output directory is in remote filesystem: {output_dir_protocol}. Failed to mkdir. Ensure the remote output directory exists."
                )

        self.model_kwargs = args["spec"].get("model_kwargs", dict())
        self.confidence_interval_width = args["spec"].get("confidence_interval_width")

    def build_model(self):
        view = dp.View(dp.Text("# My report 3"))
        start_time = time.time()
        if self.model == "prophet":
            self.data, self.models, self.outputs = prophet_operate(self)
        elif self.model == "neuralprophet":
            self.data, self.models, self.outputs = neuralprophet_operate(self)
        elif self.model == "arima":
            self.data, self.models, self.outputs = arima_operate(self)
        else:
            raise ValueError(f"Unsupported model type: {self.model}")
        self.elapsed_time = time.time() - start_time
        return self.generate_report(self.elapsed_time)

    def generate_report(self, elapsed_time):
        def get_select_plot_list(fn):
            return dp.Select(
                blocks=[
                    dp.Plot(fn(i), label=col)
                    for i, col in enumerate(self.target_columns)
                ]
            )

        title_text = dp.Text("# Forecast Report")
        forecast_col_name = "yhat"
        ci_col_names = None
        train_metrics = True
        model_description = dp.Text("---")
        other_sections = []
        ds_forecast_col = None

        if self.model == "prophet":
            model_description = dp.Text(
                "Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well."
            )
            other_sections = get_prophet_report(self)
            ds_column_series = self.data["ds"]
            ds_forecast_col = self.outputs[0]["ds"]
            ci_col_names = ["yhat_lower", "yhat_upper"]
        elif self.model == "neuralprophet":
            model_description = dp.Text(
                "NeuralProphet is an easy to learn framework for interpretable time series forecasting. NeuralProphet is built on PyTorch and combines Neural Network and traditional time-series algorithms, inspired by Facebook Prophet and AR-Net."
            )
            other_sections = get_neuralprophet_report(self)
            forecast_col_name = "yhat1"
            ds_column_series = self.data["ds"]
            ds_forecast_col = self.outputs[0]["ds"]
        elif self.model == "arima":
            model_description = dp.Text(
                "An autoregressive integrated moving average, or ARIMA, is a statistical analysis model that uses time series data to either better understand the data set or to predict future trends. A statistical model is autoregressive if it predicts future values based on past values."
            )
            other_sections = get_arima_report(self)
            train_metrics = False
            ds_column_series = self.data[self.ds_column]
            ds_forecast_col = self.outputs[0].index
            ci_col_names = ["yhat_lower", "yhat_upper"]

        md_columns = " * ".join([f"{x} \n" for x in self.target_columns])
        summary = dp.Blocks(
            dp.Select(
                blocks=[
                    dp.Group(
                        dp.Text(f"You selected the **`{self.model}`** model."),
                        model_description,
                        dp.Text(
                            f"Based on your dataset, you could have also selected any of the models: `{'`, `'.join(AVAILABLE_MODELS)}`."
                        ),
                        dp.Group(
                            dp.BigNumber(
                                heading="Analysis was completed in (sec)",
                                value=int(elapsed_time),
                            ),
                            dp.BigNumber(
                                heading="Starting time index",
                                value=ds_column_series.min().strftime(
                                    "%B %d, %Y"
                                ),  # "%r" # TODO: Figure out a smarter way to format
                            ),
                            dp.BigNumber(
                                heading="Ending time index",
                                value=ds_column_series.max().strftime(
                                    "%B %d, %Y"
                                ),  # "%r" # TODO: Figure out a smarter way to format
                            ),
                            dp.BigNumber(
                                heading="Num series", value=len(self.target_columns)
                            ),
                            columns=4,
                        ),
                        dp.Text("### First 10 Rows of Data"),
                        dp.DataTable(self.original_user_data.head(10), caption="Start"),
                        dp.Text("----"),
                        dp.Text("### Last 10 Rows of Data"),
                        dp.DataTable(self.original_user_data.tail(10), caption="End"),
                        dp.Text("### Data Summary Statistics"),
                        dp.DataTable(
                            self.original_user_data.describe(),
                            caption="Summary Statistics",
                        ),
                        label="Summary",
                    ),
                    dp.Text(
                        f"The following report compares a variety of metrics and plots for your target columns: \n {md_columns}.\n",
                        label="Target Columns",
                    ),
                ]
            ),
        )

        train_metric_sections = []
        if train_metrics:
            self.eval_metrics = evaluate_metrics(
                self.target_columns,
                self.data,
                self.outputs,
                target_col=forecast_col_name,
            )
            sec6_text = dp.Text(f"## Historical Data Evaluation Metrics")
            sec6 = dp.DataTable(self.eval_metrics)
            train_metric_sections = [sec6_text, sec6]

        test_eval_metrics = []
        test_data = None
        if self.test_filename:
            self.test_eval_metrics, summary_metrics, test_data = test_evaluate_metrics(
                self.target_columns,
                self.test_filename,
                self.outputs,
                self,
                target_col=forecast_col_name,
            )
            sec7_text = dp.Text(f"## Holdout Data Evaluation Metrics")
            sec7 = dp.DataTable(self.test_eval_metrics)

            sec8_text = dp.Text(f"## Holdout Data Summary Metrics")
            sec8 = dp.DataTable(summary_metrics)

            test_eval_metrics = [sec7_text, sec7, sec8_text, sec8]

        forecast_text = dp.Text(f"## Forecasted Data Overlaying Historical")
        forecast_sec = get_forecast_plots(
            self.data,
            self.outputs,
            self.target_columns,
            test_data=test_data,
            forecast_col_name=forecast_col_name,
            ds_col=ds_column_series,
            ds_forecast_col=ds_forecast_col,
            ci_col_names=ci_col_names,
            ci_interval_width=self.confidence_interval_width,
        )
        forecast_plots = [forecast_text, forecast_sec]

        yaml_appendix_title = dp.Text(f"## Reference: YAML File")
        yaml_appendix = dp.Code(code=yaml.dump(self.args), language="yaml")
        all_sections = (
            [title_text, summary]
            + forecast_plots
            + other_sections
            + test_eval_metrics
            + train_metric_sections
            + [yaml_appendix_title, yaml_appendix]
        )
        self.view = dp.View(*all_sections)
        dp.save_report(self.view, self.report_file_name, open=True)
        print(f"Generated Report: {self.report_file_name}.")
        return


def operate(args):
    operator = ForecastOperator(args).build_model()
    return operator


def run():
    args = json.loads(os.environ.get("OPERATOR_ARGS", "{}"))
    return operate(args)


if __name__ == "__main__":
    run()
