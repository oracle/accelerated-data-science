#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import logging
import os
import sys
from ads.jobs.builders.runtimes.python_runtime import PythonRuntime
import datapane as dp
from prophet.plot import add_changepoints_to_plot
from prophet import Prophet
# from neuralprophet import NeuralProphet as Prophet
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
from ads.operators.forecast.utils import mape
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
        self.model = args["forecast"]["model"]
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

    def load_data(self):
        # Load data and format datetime column
        pd_format = self.historical_data.get("format")
        if not pd_format:
            _, pd_format = os.path.splitext(self.input_filename)
            pd_format = pd_format[1:]
        if pd_format in ["json", "clipboard", "excel", "csv", "feather", "hdf"]:
            read_fn = getattr(pd, f"read_{pd_format}")
            data = read_fn(self.input_filename, storage_options=self.storage_options)
            # if columns:
            #     # keep only these columns, done after load because only CSV supports stream filtering
            #     df = df[columns]
        else:
            raise ValueError(f"Unrecognized format: {pd_format}")

        data["ds"] = pd.to_datetime(data[self.ds_column], format=self.datetime_format)
        data.drop([self.ds_column], axis=1, inplace=True)
        data.fillna(0, inplace=True)
        self.data = data

        models = []
        outputs = []
        for i, col in enumerate(self.target_columns):
            data_i = data[[col, "ds"]]
            print(f"using columns: {data_i.columns}")
            data_i.rename({col:"y"}, axis=1, inplace=True)
            
            model = Prophet()
            model.fit(data_i)

            future = model.make_future_dataframe(periods=self.horizon['periods']) #, freq=self.horizon['interval_unit']
            forecast = model.predict(future)

            print(f"-----------------Model {i}----------------------")
            print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
            models.append(model)
            outputs.append(forecast)
        
        self.models = models
        self.outputs = outputs

        print("===========Done===========")
        output_total = pd.concat(self.outputs)
        
        output_total.to_csv(self.output_filename, storage_options=self.storage_options)
        return self.outputs

    def evaluate(self):
        total_metrics = pd.DataFrame()

        for idx, col in enumerate(self.target_columns):
            metrics = dict()
            y_true = np.asarray(self.data[col])
            y_pred = np.asarray(self.outputs[idx]["yhat"][:len(y_true)])

            metrics["MAPE"] = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
            metrics["RMSE"] = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))
            metrics["r2"] = r2_score(y_true=y_true, y_pred=y_pred)
            metrics["Explained Variance"] = explained_variance_score(y_true=y_true, y_pred=y_pred)
            
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=[col])
            total_metrics = pd.concat([total_metrics, metrics_df], axis=1)
        self.eval_metrics = total_metrics
        
    def generate_report(self):
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

        self.evaluate()
        sec6_text = dp.Text(f"## Evaluation Metrics")
        sec6 = dp.Table(self.eval_metrics)

        yaml_appendix_title = dp.Text(f"## Reference: YAML File")
        yaml_appendix = dp.Code(code=yaml.dump(self.args), language="yaml")

        self.view = dp.View(title_text, summary, sec1_text, sec1, sec2_text, sec2, sec3_text, sec3, sec4_text, sec4, sec5_text, sec5, sec6_text, sec6, yaml_appendix_title, yaml_appendix)
        dp.save_report(self.view, self.report_file_name, open=True)
        print(f"Generated Report: {self.report_file_name}")
        return self.view


def operate(args):
    operator = ForecastOperator(args)
    forecasts = operator.load_data()
    report = operator.generate_report()
    return operator


def run():
    
    args = json.loads(os.environ.get("OPERATOR_ARGS", "{}"))
    return operate(args)


if __name__ == '__main__':
    run()