#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import logging
import os
import sys
from ads.jobs.builders.runtimes.python_runtime import PythonRuntime
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
    from neuralprophet import NeuralProphet

    operator = load_data_dict(operator)
    full_data_dict = operator.full_data_dict

    models = []
    outputs = dict()
    outputs_legacy = []

    # Extract the Confidence Interval Width and convert to neural prophets equivalent - quantiles
    model_kwargs = operator.model_kwargs
    if operator.confidence_interval_width is None:
        quantiles = operator.model_kwargs.get("quantiles", [0.05, 0.95])
        operator.confidence_interval_width = float(quantiles[1]) - float(quantiles[0])
    else:
        boundaries = round((1 - operator.confidence_interval_width) / 2, 2)
        quantiles = [boundaries, operator.confidence_interval_width + boundaries]
    model_kwargs["quantiles"] = quantiles

    for i, (target, df) in enumerate(full_data_dict.items()):
        # format the dataframe for this target. Dropping NA on target[df] will remove all future data
        df = _preprocess_prophet(df, operator.ds_column, operator.datetime_format)
        data_i = df[df[target].notna()]
        data_i.rename({target: "y"}, axis=1, inplace=True)

        # Assume that all columns passed in should be used as additional data
        additional_regressors = set(data_i.columns) - {"y", "ds"}
        training_data = data_i[["y", "ds"] + list(additional_regressors)]

        # Build and fit model
        model = NeuralProphet(**model_kwargs)
        for add_reg in additional_regressors:
            model = model.add_future_regressor(name=add_reg)
        model.fit(training_data, freq=operator.horizon["interval_unit"])

        # Determine which regressors were accepted
        accepted_regressors = list(model.config_regressors.keys())
        print(f"Found the following additional data columns: {additional_regressors}")
        print(
            f"While fitting the model, some additional data may have been discarded. Only using the columns: {accepted_regressors}"
        )

        # Build future dataframe
        future = df.reset_index(drop=True)
        future["y"] = None
        future = future[["y", "ds"] + list(accepted_regressors)]

        # Forecaset model and collect outputs
        forecast = model.predict(future)
        print(f"-----------------Model {i}----------------------")
        print(forecast.tail())
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
        for cat in operator.categories:  # Note: to restrict columns, set this to [:2]
            output_i = pd.DataFrame()

            output_i[operator.ds_column] = outputs[f"{col}_{cat}"]["ds"]
            output_i[operator.target_category_column] = cat
            output_i[f"{col}_forecast"] = outputs[f"{col}_{cat}"]["yhat1"]
            output_i[f"{col}_forecast_upper"] = outputs[f"{col}_{cat}"][
                f"yhat1 {quantiles[1]*100}%"
            ]
            output_i[f"{col}_forecast_lower"] = outputs[f"{col}_{cat}"][
                f"yhat1 {quantiles[0]*100}%"
            ]
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


def get_neuralprophet_report(self):
    import datapane as dp

    def get_select_plot_list(fn):
        return dp.Select(
            blocks=[
                dp.Plot(fn(i), label=col) for i, col in enumerate(self.target_columns)
            ]
        )

    sec1_text = dp.Text(
        f"## Forecast Overview \nThese plots show your forecast in the context of historical data."
    )  # TODO add confidence intervals
    sec1 = get_select_plot_list(lambda idx: self.models[idx].plot(self.outputs[idx]))

    sec2_text = dp.Text(f"## Forecast Broken Down by Trend Component")
    sec2 = get_select_plot_list(
        lambda idx: self.models[idx].plot_components(self.outputs[idx])
    )

    sec3_text = dp.Text(f"## Forecast Parameter Plots")
    sec3 = get_select_plot_list(lambda idx: self.models[idx].plot_parameters())

    # Auto-corr
    # sec4_text = dp.Text(f"## Auto-Correlation Plots")
    # output_series = []
    # for idx in range(len(self.target_columns)):
    #     series = pd.Series(self.outputs[idx]["yhat1"])
    #     series.index = pd.DatetimeIndex(self.outputs[idx]["ds"])
    #     output_series.append(series)
    # sec4 = get_select_plot_list(lambda idx: pd.plotting.autocorrelation_plot(output_series[idx]))

    sec5_text = dp.Text(f"## Neural Prophet Model Parameters")
    model_states = []
    for i, m in enumerate(self.models):
        model_states.append(
            pd.Series(
                m.state_dict(), index=m.state_dict().keys(), name=self.target_columns[i]
            )
        )
    all_model_states = pd.concat(model_states, axis=1)
    sec5 = dp.DataTable(all_model_states)

    # return [sec4_text, sec4]
    return [sec1_text, sec1, sec2_text, sec2, sec3_text, sec3, sec5_text, sec5]


# from neuralprophet import NeuralProphet
# import pandas as pd
# df = pd.read_csv('pypistats.csv')
# df['ds'] = df['date']
# df['y'] = df['ocifs_downloads']
# df = df[['ds', 'y']]
# m = NeuralProphet()
# metrics = m.fit(df, freq="D")
# forecast = m.predict(df)
