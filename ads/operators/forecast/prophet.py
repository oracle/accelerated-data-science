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
import pandas as pd
from ads.operators.forecast.utils import load_data_dict, _write_data

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def _preprocess_prophet(data, ds_column, datetime_format):
    data["ds"] = pd.to_datetime(data[ds_column], format=datetime_format)
    return data.drop([ds_column], axis=1)


def operate(operator):
    # Load in the data as a dictionary of {target: dataset}
    operator = load_data_dict(operator)
    full_data_dict = operator.full_data_dict
    models = []
    outputs = dict()
    outputs_legacy = []

    # Extract the Confidence Interval Width and convert to prophet's equivalent - interval_width
    if operator.confidence_interval_width is None:
        operator.confidence_interval_width = operator.model_kwargs.get(
            "interval_width", 0.90
        )
    model_kwargs = operator.model_kwargs
    model_kwargs["interval_width"] = operator.confidence_interval_width

    for i, (target, df) in enumerate(full_data_dict.items()):
        # format the dataframe for this target. Dropping NA on target[df] will remove all future data
        df = _preprocess_prophet(df, operator.ds_column, operator.datetime_format)
        data_i = df[df[target].notna()]
        data_i.rename({target: "y"}, axis=1, inplace=True)

        # Assume that all columns passed in should be used as additional data
        additional_regressors = set(data_i.columns) - {"y", "ds"}
        print(f"Found the following additional data columns: {additional_regressors}")

        # Build and fit model
        model = Prophet(**model_kwargs)
        for add_reg in additional_regressors:
            model.add_regressor(add_reg)
        model.fit(data_i)

        # Make future df for prediction
        if len(additional_regressors):
            # TOOD: this will use the period/range of the additional data
            future = df.drop(target, axis=1)
        else:
            future = model.make_future_dataframe(
                periods=operator.horizon["periods"],
                freq=operator.horizon["interval_unit"],
            )
        # Make Prediction
        forecast = model.predict(future)
        print(f"-----------------Model {i}----------------------")
        print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

        # Collect Outputs
        models.append(model)
        outputs[target] = forecast
        outputs_legacy.append(forecast)

    operator.models = models
    operator.outputs = outputs_legacy

    print("===========Done===========")
    outputs_merged = pd.DataFrame()

    # Merge the outputs from each model into 1 df with all outputs by target and category
    for col in operator.original_target_columns:
        output_col = pd.DataFrame()
        for cat in operator.categories:  # Note: add [:2] to restrict
            output_i = pd.DataFrame()

            output_i[operator.ds_column] = outputs[f"{col}_{cat}"]["ds"]
            output_i[operator.target_category_column] = cat
            output_i[f"{col}_forecast"] = outputs[f"{col}_{cat}"]["yhat"]
            output_i[f"{col}_forecast_upper"] = outputs[f"{col}_{cat}"]["yhat_upper"]
            output_i[f"{col}_forecast_lower"] = outputs[f"{col}_{cat}"]["yhat_lower"]
            output_col = pd.concat([output_col, output_i])
        output_col = output_col.sort_values(operator.ds_column).reset_index(drop=True)
        outputs_merged = pd.concat([outputs_merged, output_col], axis=1)
    _write_data(
        outputs_merged, operator.output_filename, "csv", operator.storage_options
    )

    # Re-merge historical datas for processing
    data_merged = pd.concat(
        [v[v[k].notna()].set_index("ds") for k, v in full_data_dict.items()], axis=1
    ).reset_index()

    return data_merged, models, outputs_legacy


def get_prophet_report(self):
    def get_select_plot_list(fn):
        return dp.Select(
            blocks=[
                dp.Plot(fn(i), label=col) for i, col in enumerate(self.target_columns)
            ]
        )

    sec1_text = dp.Text(
        f"## Forecast Overview \nThese plots show your forecast in the context of historical data."
    )
    sec1 = get_select_plot_list(
        lambda idx: self.models[idx].plot(self.outputs[idx], include_legend=True)
    )

    sec2_text = dp.Text(f"## Forecast Broken Down by Trend Component")
    sec2 = get_select_plot_list(
        lambda idx: self.models[idx].plot_components(self.outputs[idx])
    )

    sec3_text = dp.Text(f"## Forecast Changepoints")
    sec3_figs = [
        self.models[idx].plot(self.outputs[idx])
        for idx in range(len(self.target_columns))
    ]
    [
        add_changepoints_to_plot(
            sec3_figs[idx].gca(), self.models[idx], self.outputs[idx]
        )
        for idx in range(len(self.target_columns))
    ]
    sec3 = get_select_plot_list(lambda idx: sec3_figs[idx])

    # # Auto-corr
    # sec4_text = dp.Text(f"## Auto-Correlation Plots")
    # auto_corr_figures = []
    # for idx in range(len(self.target_columns)):
    #     series = pd.Series(self.outputs[idx]["yhat"])
    #     series.index = pd.DatetimeIndex(self.outputs[idx]["ds"])
    #     auto_corr_figures.append(pd.plotting.autocorrelation_plot(series).figure)
    # sec4 = get_select_plot_list(lambda idx: auto_corr_figures[idx])

    all_sections = [sec1_text, sec1, sec2_text, sec2, sec3_text, sec3]

    sec5_text = dp.Text(f"## Prophet Model Seasonality Components")
    model_states = []
    for i, m in enumerate(self.models):
        model_states.append(
            pd.Series(
                m.seasonalities,
                index=m.seasonalities.keys(),
                name=self.target_columns[i],
            )
        )
    all_model_states = pd.concat(model_states, axis=1)
    if not all_model_states.empty:
        sec5 = dp.DataTable(all_model_states)
        all_sections = all_sections + [sec5_text, sec5]

    return all_sections  # + [sec4_text, sec4]


# from sklearn.model_selection import TimeSeriesSplit
