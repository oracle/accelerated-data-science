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
from prophet import Prophet
import pandas as pd
from urllib.parse import urlparse
import json
import yaml


import ads
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import oci
import time
from datetime import datetime
from ads.operators.forecast.prophet import operate as prophet_operate
# from ads.operators.forecast.neural_prophet import operate as neuralprophet_operate
from ads.operators.forecast.utils import evaluate_metrics
from sklearn.metrics import mean_absolute_percentage_error, explained_variance_score, r2_score, mean_squared_error
from sklearn.datasets import load_files

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class ForecastOperator:
    def __init__(self, args):
        self.args = args
        assert args['kind'] == "operator"
        assert args['type'] == "forecast"
        assert args["version"] == "1"
        self.historical_data = args["historical_data"]
        self.output_data = args["output_data"]
        self.model = args["forecast"]["model"].lower()
        self.target_columns = args["forecast"]["target_columns"]
        self.datetime_column = args["forecast"]["datetime_column"]
        self.horizon = args["forecast"]["horizon"]
        self.report_file_name = args["forecast"]["report_file_name"]
        
        # TODO: clean up
        self.input_filename = self.historical_data["url"]
        self.output_filename = self.output_data["url"]
        self.ds_column = self.datetime_column.get("name")
        self.datetime_format = self.datetime_column.get("format")
        self.storage_options = {
            "profile": self.args['execution'].get('oci_profile'),
            "config": self.args['execution'].get('oci_config'),
        }

    def build_model(self):
        if self.model == "prophet":
            self.data, self.models, self.outputs = prophet_operate(self)
            return self.generate_prophet_report()
        # elif self.model == "neuralprophet":
        #     operator = neuralprophet_operate(self)
        #     return operator
        elif self.model == "arima":
            raise NotImplementedError()
        raise ValueError(f"Unsupported model type: {self.model}")
        
    def generate_prophet_report(self):
        def get_select_plot_list(fn):
            return dp.Select(blocks=[dp.Plot(fn(i), label=col) for i, col in enumerate(self.target_columns)])

        title_text = dp.Text("# Forecast Report")
        summary = dp.Text(f"You selected the `prophet` model. Based on your dataset, you could have also selected the `neuralprophet` and `ARIMA` models. The following report compares a variety of metrics and plots for your target columns: `{'`, `'.join(self.target_columns)}`.")

        sec1_text = dp.Text(f"## Forecast Overview \nThese plots show your forecast in the context of historical data with 80% confidence.")
        sec1 = get_select_plot_list(lambda idx: self.models[idx].plot(self.outputs[idx], include_legend=True))

        sec2_text = dp.Text(f"## Forecast Broken Down by Trend Component")
        sec2 = get_select_plot_list(lambda idx: self.models[idx].plot_components(self.outputs[idx]))

        sec3_text = dp.Text(f"## Forecast Changepoints")
        sec3_figs = [self.models[idx].plot(self.outputs[idx]) for idx in range(len(self.target_columns))]
        [add_changepoints_to_plot(sec3_figs[idx].gca(), self.models[idx], self.outputs[idx]) for idx in range(len(self.target_columns))]
        sec3 = get_select_plot_list(lambda idx: sec3_figs[idx])

        # Auto-corr
        sec4_text = dp.Text(f"## Auto-Correlation Plots")
        # output_series = [pd.DataFrame(self.outputs[idx][["yhat", "ds"]]).set_index("ds")["yhat"] for idx in range(len(self.target_columns))]
        output_series = [pd.Series(self.outputs[idx]["yhat"]) for idx in range(len(self.target_columns))]
        sec4 = get_select_plot_list(lambda idx: pd.plotting.autocorrelation_plot(output_series[idx]))

        sec5_text = dp.Text(f"## Forecast Seasonality Parameters")
        sec5 = dp.Select(blocks=[dp.Table(pd.DataFrame(m.seasonalities), label=self.target_columns[i]) for i, m in enumerate(self.models)])

        self.eval_metrics = evaluate_metrics(self.target_columns, self.data, self.outputs)
        sec6_text = dp.Text(f"## Evaluation Metrics")
        sec6 = dp.Table(self.eval_metrics)

        yaml_appendix_title = dp.Text(f"## Reference: YAML File")
        yaml_appendix = dp.Code(code=yaml.dump(self.args), language="yaml")

        self.view = dp.View(title_text, summary, sec1_text, sec1, sec2_text, sec2, sec3_text, sec3, sec4_text, sec4, sec5_text, sec5, sec6_text, sec6, yaml_appendix_title, yaml_appendix)
        dp.save_report(self.view, self.report_file_name, open=True)
        print(f"Generated Report: {self.report_file_name}")
        return


def operate(args):
    operator = ForecastOperator(args).build_model()
    return operator


def run():
    args = json.loads(os.environ.get("OPERATOR_ARGS", "{}"))
    return operate(args)


if __name__ == '__main__':
    run()