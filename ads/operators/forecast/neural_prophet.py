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
from neuralprophet import NeuralProphet as Prophet
import pandas as pd

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class ForecastOperator:
    def __init__(self, **kwargs):
        self.input_filename = "pypistats.csv"
        self.report_filename = "report.html"
        self.output_filename = "output.csv"
        self.ds_column = "date"
        self.datetime_format = None
        self.target_columns = ["ocifs_downloads", "oracle-ads_downloads", "oci-mlflow_downloads"]

        self.horizon = {
            "periods": 31,
            "interval": 1,
            "interval_unit": "D",
        }

    def load_data(self):
        # Load data and format datetime column
        data = pd.read_csv(self.input_filename)
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
            # Add regressors
            # Add metrics
            # Use forecasting service datasets
            # report should have html colored code for yaml file
            model.fit(data_i)

            future = model.make_future_dataframe(periods=self.horizon['periods']) #, freq=self.horizon['interval_unit']
            forecast = model.predict(future)

            print(f"-------Model {i}----------------------")
            print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
            models.append(model)
            outputs.append(forecast)
        
        self.models = models
        self.outputs = outputs

        print("===========Done===========")
        output_total = pd.concat(self.outputs).to_csv(self.output_filename)
        return self.outputs

    def generate_report(self):
        def get_select_plot_list(fn):
            return dp.Select(blocks=[dp.Plot(fn(i), label=col) for i, col in enumerate(self.target_columns)])

        title_text = dp.Text("# Forecast Report")
        sec1_text = dp.Text(f"## Forecast Overview")
        sec1 = get_select_plot_list(lambda idx: self.models[idx].plot(self.outputs[idx], include_legend=True))
        sec2_text = dp.Text(f"## Forecast Broken Down by Trend Component")
        sec2 = get_select_plot_list(lambda idx: self.models[idx].plot_components(self.outputs[idx]))

        sec3_text = dp.Text(f"## Forecast Changepoints")
        sec3_figs = [self.models[idx].plot(self.outputs[idx]) for idx in range(len(self.target_columns))]
        [add_changepoints_to_plot(sec3_figs[idx].gca(), self.models[idx], self.outputs[idx]) for idx in range(len(self.target_columns))]
        sec3 = get_select_plot_list(lambda idx: sec3_figs[idx])
        sec4_text = dp.Text(f"## Forecast Seasonality Parameters")
        sec4 = dp.Select(blocks=[dp.Table(pd.DataFrame(m.seasonalities), label=self.target_columns[i]) for i, m in enumerate(self.models)])

        self.view = dp.View(title_text, sec1_text, sec1, sec2_text, sec2, sec3_text, sec3, sec4_text, sec4)
        dp.save_report(self.view, self.report_filename, open=True)
        print(f"Generated Report: {self.report_filename}")
        return self.view


def operate(args):
    operator = ForecastOperator(**args)
    forecasts = operator.load_data()
    report = operator.generate_report()
    return forecasts

    # Return fully verbose yaml
    # Offer some explanations
    # Reccomend other possible models

if __name__ == '__main__':
    operate(dict())
