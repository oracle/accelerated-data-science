#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
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
import report_creator as rc


def _label_encode_dataframe(df, no_encode=set()):
    df_to_encode = df[list(set(df.columns) - no_encode)]
    le = DataFrameLabelEncoder().fit(df_to_encode)
    return le, le.transform(df)


def _inverse_transform_dataframe(le, df):
    return le.inverse_transform(df)


def smape(actual, predicted) -> float:
    if not all([isinstance(actual, np.ndarray), isinstance(predicted, np.ndarray)]):
        actual, predicted = (np.array(actual), np.array(predicted))
    zero_mask = np.logical_and(actual == 0, predicted == 0)

    denominator = np.abs(actual) + np.abs(predicted)
    denominator[zero_mask] = 1

    numerator = np.abs(actual - predicted)
    default_output = np.ones_like(numerator) * np.inf

    abs_error = np.divide(numerator, denominator)
    return round(np.mean(abs_error) * 100, 2)


def _build_metrics_per_horizon(
    test_data: "TestData",
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

    test_df = (
        test_data.get_data_long()
        .rename({test_data.dt_column_name: ForecastOutputColumns.DATE}, axis=1)
        .set_index([ForecastOutputColumns.DATE, ForecastOutputColumns.SERIES])
        .sort_index()
    )
    forecast_df = (
        output.get_horizon_long()
        .set_index([ForecastOutputColumns.DATE, ForecastOutputColumns.SERIES])
        .sort_index()
    )

    dates = test_df.index.get_level_values(0).unique()
    common_idx = test_df.index.intersection(forecast_df.index)

    if len(common_idx) != len(forecast_df.index):
        if len(dates) > output.horizon:
            logger.debug(
                f"Found more unique dates ({len(dates)}) in the Test Data than expected given the horizon ({output.horizon})."
            )
        elif len(dates) < output.horizon:
            logger.debug(
                f"Found fewer unique dates ({len(dates)}) in the Test Data than expected given the horizon ({output.horizon}). This will impact the metrics."
            )
        elif test_df.index.get_level_values(1).unique() > output.list_series_ids():
            logger.debug(
                f"Found more Series Ids in test data ({len(dates)}) expected from the historical data ({output.list_series_ids()})."
            )
        else:
            logger.debug(
                f"Found fewer Series Ids in test data ({len(dates)}) expected from the historical data ({output.list_series_ids()}). This will impact the metrics."
            )

    test_df = test_df.loc[common_idx]
    forecast_df = forecast_df.loc[common_idx]

    totals = test_df.sum(numeric_only=True)
    wmape_weights = np.array((totals / totals.sum()).values)

    metrics_df = pd.DataFrame()
    for date in dates:
        y_true = test_df.xs(date, level=ForecastOutputColumns.DATE)[
            test_data.target_name
        ]
        y_pred = forecast_df.xs(date, level=ForecastOutputColumns.DATE)[
            ForecastOutputColumns.FORECAST_VALUE
        ]
        y_true = np.array(y_true.values)
        y_pred = np.array(y_pred.values)

        drop_na_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if not drop_na_mask.all():  # There is a missing value
            if drop_na_mask.any():  # All values are missing
                logger.debug(
                    f"No test data available for date: {date}. This will affect the test metrics."
                )
                continue
            logger.debug(
                f"Missing test data for date: {date}. This will affect the test metrics."
            )
            y_true = y_true[drop_na_mask]
            y_pred = y_pred[drop_na_mask]
        smapes = smape(actual=y_true, predicted=y_pred)
        mapes = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
        wmapes = mapes * wmape_weights

        metrics_df = pd.concat(
            [
                metrics_df,
                pd.DataFrame(
                    {
                        SupportedMetrics.MEAN_SMAPE: np.mean(smapes),
                        SupportedMetrics.MEDIAN_SMAPE: np.median(smapes),
                        SupportedMetrics.MEAN_MAPE: np.mean(mapes),
                        SupportedMetrics.MEDIAN_MAPE: np.median(mapes),
                        SupportedMetrics.MEAN_WMAPE: np.mean(wmapes),
                        SupportedMetrics.MEDIAN_WMAPE: np.median(wmapes),
                    },
                    index=[date],
                ),
            ]
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


def _build_metrics_df(y_true, y_pred, series_id):
    if len(y_true) == 0 or len(y_pred) == 0:
        return pd.DataFrame()
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
    return pd.DataFrame.from_dict(metrics, orient="index", columns=[series_id])


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
            ]
            forecast_by_s_id = forecast_by_s_id.dropna()
            y_true = forecast_by_s_id["input_value"].values
            y_pred = forecast_by_s_id["fitted_value"].values
            drop_na_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
            if not drop_na_mask.all():  # There is a missing value
                if drop_na_mask.any():  # All values are missing
                    logger.debug(
                        f"No fitted values available for series: {s_id}. This will affect the training metrics."
                    )
                    continue
                logger.debug(
                    f"Missing fitted values for series: {s_id}. This will affect the training metrics."
                )
                y_true = y_true[drop_na_mask]
                y_pred = y_pred[drop_na_mask]
            metrics_df = _build_metrics_df(
                y_true=y_true,
                y_pred=y_pred,
                series_id=s_id,
            )
            total_metrics = pd.concat([total_metrics, metrics_df], axis=1)
        except Exception as e:
            logger.debug(
                f"Failed to generate training metrics for target_series: {s_id}"
            )
            logger.debug(f"Recieved Error Statement: {e}")
    return total_metrics


def _select_plot_list(fn, series_ids):
    blocks = [rc.Widget(fn(s_id=s_id), label=s_id) for s_id in series_ids]
    return rc.Select(blocks=blocks) if len(blocks) > 1 else blocks[0]


def _add_unit(num, unit):
    return f"{num} {unit}"

def get_auto_select_plot(backtest_results):
    fig = go.Figure()
    columns = backtest_results.columns.tolist()
    back_test_column = "backtest"
    columns.remove(back_test_column)
    for i, column in enumerate(columns):
        color = 0 #int(i * 255 / len(columns))
        fig.add_trace(
            go.Scatter(
            x=backtest_results[back_test_column],
            y=backtest_results[column],
            mode="lines",
            name=column,
        ))

    return rc.Widget(fig)


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
    if target_col is not None and target_col != "":
        temp = target_col + "_"
        if temp in target:
            target = target.replace(temp, "", 1)
    return target


def default_signer(**kwargs):
    os.environ["EXTRA_USER_AGENT_INFO"] = "Forecast-Operator"
    from ads.common.auth import default_signer

    return default_signer(**kwargs)
