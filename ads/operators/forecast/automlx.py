#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import sys
import traceback

import datapane as dp
import numpy as np
import pandas as pd

from ads.operators.forecast.utils import load_data_dict, _write_data

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def operate(operator):
    try:
        import automl
        from automl import init
    except Exception as ex:
        print(
            "Please run `pip3 install \
            --extra-index-url=https://artifacthub-phx.oci.oraclecorp.com/artifactory/api/pypi/automlx-pypi/simple \
            automlx==23.2.1` to install the required dependencies for automlx."
        )
        logger.debug(ex)
        logger.debug(traceback.format_exc())
        exit()
    operator = load_data_dict(operator)
    full_data_dict = operator.full_data_dict
    models = dict()
    outputs = dict()
    outputs_legacy = []
    selected_models = dict()
    date_column = operator.datetime_column['name']
    for i, (target, df) in enumerate(full_data_dict.items()):
        print("Running automl for {} at position {}".format(target, i))
        series_values = df[df[target].notna()]
        # drop NaNs for the time period where data wasn't recorded
        series_values.dropna(inplace=True)
        df[date_column] = pd.to_datetime(df[date_column])
        y = df.set_index(date_column)
        y_train = y
        print("Time Index is", "" if y.index.is_monotonic else "NOT", "monotonic.")
        model = automl.Pipeline(task='forecasting', n_algos_tuned=4)
        model.fit(X=None, y=y_train)
        print('Selected model: {}'.format(model.selected_model_))
        print('Selected model params: {}'.format(model.selected_model_params_))
        summary_frame = model.forecast(periods=operator.horizon["periods"],
                                       alpha=1 - (operator.confidence_interval_width / 100))
        # Collect Outputs
        selected_models[target] = {"series_id": target, "selected_model": model.selected_model_,
                                   "model_params": model.selected_model_params_}
        models[target] = model
        summary_frame = summary_frame.rename_axis("ds").reset_index()
        summary_frame = summary_frame.rename(
            columns={f"{target}_ci_upper": 'yhat_upper', f"{target}_ci_lower": 'yhat_lower', f"{target}": 'yhat'})
        # In case of Naive model, model.forecast function call does not return confidence intervals.
        if 'yhat_upper' not in summary_frame:
            summary_frame['yhat_upper'] = np.NAN
            summary_frame['yhat_lower'] = np.NAN
        outputs[target] = summary_frame
        outputs_legacy.append(summary_frame)

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
        [v[v[k].notna()].set_index(date_column) for k, v in full_data_dict.items()], axis=1
    ).reset_index()
    return data_merged, models, outputs_legacy


def get_automlx_report(operator):
    selected_models_text = dp.Text(
        f"## Selected Models Overview \n The following tables provide information regarding the chosen model for each series and the corresponding parameters of the models."
    )
    selected_models = dict()
    models = operator.models
    for i, (target, df) in enumerate(operator.full_data_dict.items()):
        selected_models[target] = {"series_id": target, "selected_model": models[target].selected_model_,
                                   "model_params": models[target].selected_model_params_}
    selected_models_df = pd.DataFrame(selected_models.items(), columns=['series_id', 'best_selected_model'])
    selected_df = selected_models_df['best_selected_model'].apply(pd.Series)
    selected_models_section = dp.Blocks("### Best Selected model ", dp.Table(selected_df))
    all_sections = [selected_models_text, selected_models_section]
    return all_sections
