#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import sys
from typing import List

import fsspec
import numpy as np
import pandas as pd
import cloudpickle
import plotly.express as px
from plotly import graph_objects as go
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_percentage_error,
    mean_squared_error,
)

try:
    from scipy.stats import linregress
except:
    from sklearn.metrics import r2_score

from ads.common.object_storage_details import ObjectStorageDetails
from ads.dataset.label_encoder import DataFrameLabelEncoder
from ads.opctl import logger

from .const import SupportedMetrics, SupportedModels, RENDER_LIMIT
from .errors import ForecastInputDataError, ForecastSchemaYamlError
from .operator_config import ForecastOperatorSpec, ForecastOperatorConfig
from ads.opctl.operator.lowcode.common.utils import merge_category_columns
from ads.opctl.operator.lowcode.forecast.const import ForecastOutputColumns

# from ads.opctl.operator.lowcode.forecast.model.forecast_datasets import TestData, ForecastOutput


def _label_encode_dataframe(df, no_encode=set()):
    df_to_encode = df[list(set(df.columns) - no_encode)]
    le = DataFrameLabelEncoder().fit(df_to_encode)
    return le, le.transform(df)


def _inverse_transform_dataframe(le, df):
    return le.inverse_transform(df)


def smape(actual, predicted) -> float:
    if not all([isinstance(actual, np.ndarray), isinstance(predicted, np.ndarray)]):
        actual, predicted = (np.array(actual), np.array(predicted))
    denominator = np.abs(actual) + np.abs(predicted)
    numerator = np.abs(actual - predicted)
    default_output = np.ones_like(numerator) * np.inf

    abs_error = np.divide(numerator, denominator)
    return round(np.mean(abs_error) * 100, 2)


def _build_metrics_per_horizon(
    data: "TestData",
    output: "ForecastOutput",
) -> pd.DataFrame:
    """
    Calculates Mean sMAPE, Median sMAPE, Mean MAPE, Median MAPE, Mean wMAPE, Median wMAPE for each horizon

    Parameters
    ------------
    data:  Pandas Dataframe
            Dataframe that has the actual data
    output: Pandas Dataframe
            Dataframe that has the forecasted data

    Returns
    --------
    Pandas Dataframe
        Dataframe with Mean sMAPE, Median sMAPE, Mean MAPE, Median MAPE, Mean wMAPE, Median wMAPE values for each horizon
    """
    """
    Assumptions:
    data and output have all the target columns.
    yhats in output are in the same order as in series_ids.
    Test data might not have sorted dates and the order of series also might differ.
    """

    actuals_df = data[data.dt_column_name, data.target_name]

    # Concat the yhats in output and include only dates that are in test data
    forecasts_df = pd.DataFrame()
    for s_id in output.list_series_ids():
        forecast_i = output.get_forecast(s_id)[["Date", "forecast_value"]]
        forecast_i = forecast_i[
            forecast_i["Date"].isin(actuals_df[data.dt_column_name])
        ]
        forecasts_df = pd.concat([forecasts_df, forecast_i.set_index("Date")], axis=1)

    # Remove dates that are not there in output
    actuals_df = actuals_df[
        actuals_df[data.dt_column_name].isin(forecasts_df.index.values)
    ]

    if actuals_df.empty or forecasts_df.empty:
        return pd.DataFrame()

    totals = actuals_df.sum(numeric_only=True)
    wmape_weights = np.array((totals / totals.sum()).values)

    actuals_df = actuals_df.set_index(data.dt_column_name)

    metrics_df = pd.DataFrame(
        columns=[
            SupportedMetrics.MEAN_SMAPE,
            SupportedMetrics.MEDIAN_SMAPE,
            SupportedMetrics.MEAN_MAPE,
            SupportedMetrics.MEDIAN_MAPE,
            SupportedMetrics.MEAN_WMAPE,
            SupportedMetrics.MEDIAN_WMAPE,
        ]
    )

    for i, (y_true, y_pred) in enumerate(
        zip(actuals_df.itertuples(index=False), forecasts_df.itertuples(index=False))
    ):
        y_true, y_pred = np.array(y_true), np.array(y_pred)

        smapes = np.array(
            [smape(actual=y_t, predicted=y_p) for y_t, y_p in zip(y_true, y_pred)]
        )
        mapes = np.array(
            [
                mean_absolute_percentage_error(y_true=[y_t], y_pred=[y_p])
                for y_t, y_p in zip(y_true, y_pred)
            ]
        )
        wmapes = np.array([mape * weight for mape, weight in zip(mapes, wmape_weights)])

        metrics_row = {
            SupportedMetrics.MEAN_SMAPE: np.mean(smapes),
            SupportedMetrics.MEDIAN_SMAPE: np.median(smapes),
            SupportedMetrics.MEAN_MAPE: np.mean(mapes),
            SupportedMetrics.MEDIAN_MAPE: np.median(mapes),
            SupportedMetrics.MEAN_WMAPE: np.mean(wmapes),
            SupportedMetrics.MEDIAN_WMAPE: np.median(wmapes),
        }

        metrics_df = pd.concat(
            [metrics_df, pd.DataFrame(metrics_row, index=[actuals_df.index[i]])],
        )

    return metrics_df


def load_pkl(filepath):
    storage_options = dict()
    if ObjectStorageDetails.is_oci_path(filepath):
        storage_options = default_signer()

    with fsspec.open(filepath, "rb", **storage_options) as f:
        return cloudpickle.load(f)
    return None


def write_pkl(obj, filename, output_dir, storage_options):
    pkl_path = os.path.join(output_dir, filename)
    with fsspec.open(
        pkl_path,
        "wb",
        **storage_options,
    ) as f:
        cloudpickle.dump(obj, f)


def _build_metrics_df(y_true, y_pred, column_name):
    metrics = dict()
    metrics["sMAPE"] = smape(actual=y_true, predicted=y_pred)
    metrics["MAPE"] = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
    metrics["RMSE"] = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))
    try:
        metrics["r2"] = linregress(y_true, y_pred).rvalue ** 2
    except:
        metrics["r2"] = r2_score(y_true=y_true, y_pred=y_pred)
    metrics["Explained Variance"] = explained_variance_score(
        y_true=y_true, y_pred=y_pred
    )
    return pd.DataFrame.from_dict(metrics, orient="index", columns=[column_name])


def evaluate_train_metrics(output, metrics_col_name=None):
    """
    Training metrics

    Parameters:
    output: ForecastOutputs

    metrics_col_name: str
            Only passed in if the series column was created artifically.
            When passed in, replaces s_id as the column name in the metrics table
    """
    total_metrics = pd.DataFrame()
    for s_id in output.list_series_ids():
        try:
            forecast_by_s_id = output.get_forecast(s_id)[
                ["input_value", "Date", "fitted_value"]
            ].dropna()
            y_true = forecast_by_s_id["input_value"].values
            y_pred = forecast_by_s_id["fitted_value"].values
            metrics_df = _build_metrics_df(
                y_true=y_true,
                y_pred=y_pred,
                column_name=metrics_col_name,
            )
            total_metrics = pd.concat([total_metrics, metrics_df], axis=1)
        except Exception as e:
            logger.warn(
                f"Failed to generate training metrics for target_series: {s_id}"
            )
            logger.debug(f"Recieved Error Statement: {e}")
    return total_metrics


def _select_plot_list(fn, series_ids):
    import datapane as dp

    blocks = [dp.Plot(fn(s_id=s_id), label=s_id) for s_id in series_ids]
    return dp.Select(blocks=blocks) if len(series_ids) > 1 else blocks[0]


def _add_unit(num, unit):
    return f"{num} {unit}"


def get_forecast_plots(
    forecast_output,
    horizon,
    test_data=None,
    ci_interval_width=0.95,
):
    def plot_forecast_plotly(s_id):
        fig = go.Figure()
        forecast_i = forecast_output.get_forecast(s_id)
        actual_length = len(forecast_i)
        if actual_length > RENDER_LIMIT:
            forecast_i = forecast_i.tail(RENDER_LIMIT)
            text = (
                f"<i>To improve rendering speed, subsampled the data from {actual_length}"
                f" rows to {RENDER_LIMIT} rows for this plot.</i>"
            )
            fig.update_layout(
                annotations=[
                    go.layout.Annotation(
                        x=0.01,
                        y=1.1,
                        xref="paper",
                        yref="paper",
                        text=text,
                        showarrow=False,
                    )
                ]
            )
        upper_bound = forecast_output.upper_bound_name
        lower_bound = forecast_output.lower_bound_name
        if upper_bound is not None and lower_bound is not None:
            fig.add_traces(
                [
                    go.Scatter(
                        x=forecast_i["Date"],
                        y=forecast_i[lower_bound],
                        mode="lines",
                        line_color="rgba(0,0,0,0)",
                        showlegend=False,
                    ),
                    go.Scatter(
                        x=forecast_i["Date"],
                        y=forecast_i[upper_bound],
                        mode="lines",
                        line_color="rgba(0,0,0,0)",
                        name=f"{ci_interval_width * 100}% confidence interval",
                        fill="tonexty",
                        fillcolor="rgba(211, 211, 211, 0.5)",
                    ),
                ]
            )
        if test_data is not None:
            try:
                test_data_s_id = test_data.get_data_for_series(s_id)
                fig.add_trace(
                    go.Scatter(
                        x=test_data_s_id[test_data.dt_column_name],
                        y=test_data_s_id[test_data.target_name],
                        mode="markers",
                        marker_color="green",
                        name="Actual",
                    )
                )
            except Exception as e:
                logger.debug(f"Unable to plot test data due to: {e.args}")

        fig.add_trace(
            go.Scatter(
                x=forecast_i["Date"],
                y=forecast_i["input_value"],
                mode="markers",
                marker_color="black",
                name="Historical",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_i["Date"],
                y=forecast_i["fitted_value"],
                mode="lines+markers",
                line_color="blue",
                name="Fitted Values",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_i["Date"],
                y=forecast_i["forecast_value"],
                mode="lines+markers",
                line_color="blue",
                name="Forecast",
            )
        )
        fig.add_vline(
            x=forecast_i["Date"][-(horizon + 1) :].values[0],
            line_width=1,
            line_dash="dash",
            line_color="gray",
        )
        return fig

    return _select_plot_list(plot_forecast_plotly, forecast_output.list_series_ids())


def select_auto_model(
    datasets: "ForecastDatasets", operator_config: ForecastOperatorConfig
) -> str:
    """
    Selects AutoMLX or Arima model based on column count.

    If the number of columns is less than or equal to the maximum allowed for AutoMLX,
    returns 'AutoMLX'. Otherwise, returns 'Arima'.

    Parameters
    ------------
    datasets:  ForecastDatasets
            Datasets for predictions

    Returns
    --------
    str
        The type of the model.
    """
    freq_in_secs = datasets.get_datetime_frequency_in_seconds()
    num_of_additional_cols = len(datasets.get_additional_data_column_names())
    row_count = datasets.get_num_rows()
    number_of_series = len(datasets.list_series_ids())
    if (
        num_of_additional_cols < 15
        and row_count < 10000
        and number_of_series < 10
        and freq_in_secs > 3600
    ):
        return SupportedModels.AutoMLX
    elif row_count < 10000 and number_of_series > 10:
        return SupportedModels.AutoTS
    elif row_count > 20000:
        return SupportedModels.NeuralProphet
    else:
        return SupportedModels.NeuralProphet


def convert_target(target: str, target_col: str):
    """
    Removes the target_column that got appended to target.

    Parameters
    ------------
    target: str
        value in target_columns. i.e., "Sales_Product_Category_117"

    target_col: str
        target_column provided in yaml. i.e., "Sales"

    Returns
    --------
        Original target. i.e., "Product_Category_117"
    """
    if target_col is not None and target_col!='':
        temp = target_col + '_'
        if temp in target:
            target = target.replace(temp, '', 1)
    return target


def default_signer(**kwargs):
    os.environ["EXTRA_USER_AGENT_INFO"] = "Forecast-Operator"
    from ads.common.auth import default_signer

    return default_signer(**kwargs)
