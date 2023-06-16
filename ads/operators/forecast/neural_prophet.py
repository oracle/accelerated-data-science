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
    outputs = []
    for i, (target, df) in enumerate(full_data_dict.items()):

        df = _preprocess_prophet(df, operator.ds_column, operator.datetime_format)
        data_i = df[df[target].notna()]
        data_i.rename({target: "y"}, axis=1, inplace=True)

        additional_regressors = set(data_i.columns) - {"y", "ds"}
        illegal_columns = [
            (data_i[ar][0] == data_i[ar]).all() for ar in additional_regressors
        ]
        additional_regressors = additional_regressors - set(illegal_columns)

        training_data = data_i[["y", "ds"] + list(additional_regressors)]

        model = NeuralProphet()
        future = model.make_future_dataframe(
            df=training_data, periods=operator.horizon["periods"]
        )
        for add_reg in additional_regressors:
            model = model.add_future_regressor(name=add_reg)
        model.fit(training_data, freq=operator.horizon["interval_unit"])

        # TOOD: this will use the period/range of the additional data
        if len(additional_regressors):
            future_sorted = future.set_index("ds")
            df_sorted = df.set_index("ds")
            future = (
                future_sorted.drop(additional_regressors, axis=1)
                .join(df_sorted)
                .reset_index()
            )

        forecast = model.predict(future)

        # future_only = model.make_future_dataframe(df=data_i, periods=operator.horizon['periods']) #, freq=operator.horizon['interval_unit'])
        # all_dates = pd.concat([data_i[['ds', 'y']], future_only])
        # forecast = model.predict(all_dates)

        print(f"-----------------Model {i}----------------------")
        # forecast = forecast.rename({'yhat1': 'yhat'}, axis=1)
        print(forecast.head())
        models.append(model)
        outputs.append(forecast)

    operator.models = models
    operator.outputs = outputs

    print("===========Done===========")
    outputs_merged = outputs.copy()
    for i, col in enumerate(operator.target_columns):
        outputs_merged[i] = outputs_merged[i].rename(
            lambda x: x + "_" + col if x != "ds" else x, axis=1
        )
    output_total = pd.concat(outputs_merged, axis=1)
    _write_data(output_total, operator.output_filename, "csv", operator.storage_options)

    data_merged = pd.concat(
        [v[v[k].notna()].set_index("ds") for k, v in full_data_dict.items()], axis=1
    ).reset_index()
    return data_merged, models, outputs


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
