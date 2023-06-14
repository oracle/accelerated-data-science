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
from ads.operators.forecast.utils import evaluate_metrics, _load_data, _clean_data, _write_data

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
    data = _load_data(operator.input_filename, operator.historical_data.get("format"), operator.storage_options, columns=operator.historical_data.get("columns"))
    operator.original_user_data = data.copy()
    data = _preprocess_prophet(data, operator.ds_column, operator.datetime_format)
    data, operator.target_columns = _clean_data(data=data, 
                                                target_columns=operator.target_columns, 
                                                target_category_column=operator.target_category_column, 
                                                datetime_column="ds")
    operator.data = data
    
    models = []
    outputs = []
    for i, col in enumerate(operator.target_columns):
        data_i = data[[col, "ds"]]
        data_i.rename({col:"y"}, axis=1, inplace=True)
        
        model = NeuralProphet()
        model.fit(data_i)

        future_only = model.make_future_dataframe(df=data_i, periods=operator.horizon['periods']) #, freq=operator.horizon['interval_unit'])
        all_dates = pd.concat([data_i[['ds', 'y']], future_only])
        forecast = model.predict(all_dates)

        print(f"-----------------Model {i}----------------------")
        # forecast = forecast.rename({'yhat1': 'yhat'}, axis=1)
        print(forecast.columns)
        models.append(model)
        outputs.append(forecast)
    
    operator.models = models
    operator.outputs = outputs

    print("===========Done===========")
    outputs_merged = outputs.copy()
    for i, col in enumerate(operator.target_columns):
        outputs_merged[i] = outputs_merged[i].rename(lambda x: x+"_"+col if x != 'ds' else x, axis=1)
    output_total = pd.concat(outputs_merged, axis=1)
    _write_data(output_total, operator.output_filename, "csv", operator.storage_options)
    return data, models, outputs

def get_neuralprophet_report(self):
    import datapane as dp

    def get_select_plot_list(fn):
        return dp.Select(blocks=[dp.Plot(fn(i), label=col) for i, col in enumerate(self.target_columns)])
    
    sec1_text = dp.Text(f"## Forecast Overview \nThese plots show your forecast in the context of historical data.") # TODO add confidence intervals
    sec1 = get_select_plot_list(lambda idx: self.models[idx].plot(self.outputs[idx]))
    
    sec2_text = dp.Text(f"## Forecast Broken Down by Trend Component")
    sec2 = get_select_plot_list(lambda idx: self.models[idx].plot_components(self.outputs[idx]))
    
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
        model_states.append(pd.Series(m.state_dict(), index=m.state_dict().keys(), name=self.target_columns[i]))
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