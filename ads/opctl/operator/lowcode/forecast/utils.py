#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from ads.opctl import logger
import fsspec
import numpy as np
import pandas as pd
import json
import logging
import tempfile
import plotly.express as px
from plotly import graph_objects as go
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from typing import List
from .const import SupportedMetrics

from ads.dataset.label_encoder import DataFrameLabelEncoder
from .const import SupportedModels, MAX_COLUMNS_AUTOMLX
from .errors import ForecastInputDataError, ForecastSchemaYamlError


def _label_encode_dataframe(df, no_encode=set()):
    df_to_encode = df[list(set(df.columns) - no_encode)]
    le = DataFrameLabelEncoder().fit(df_to_encode)
    return le, le.transform(df)


def _inverse_transform_dataframe(le, df):
    return le.inverse_transform(df)


def smape(actual, predicted) -> float:
    if not all([isinstance(actual, np.ndarray), isinstance(predicted, np.ndarray)]):
        actual, predicted = (np.array(actual), np.array(predicted))
    return round(
        np.mean(np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted)))
        * 100,
        2,
    )


def _build_metrics_per_horizon(
    data: pd.DataFrame,
    outputs: pd.DataFrame,
    target_columns: List[str],
    target_col: str,
    horizon_periods: int,
) -> pd.DataFrame:
    """
    Calculates Mean sMAPE, Median sMAPE, Mean MAPE, Median MAPE, Mean wMAPE, Median wMAPE for each horizon

    Parameters
    ------------
    data:  Pandas Dataframe
            Dataframe that has the actual data
    outputs: Pandas Dataframe
            Dataframe that has the forecasted data
    target_columns: List
            List of target category columns
    target_col: str
            Target column name (yhat)
    horizon_periods: int
            Horizon Periods

    Returns
    --------
    Pandas Dataframe
        Dataframe with Mean sMAPE, Median sMAPE, Mean MAPE, Median MAPE, Mean wMAPE, Median wMAPE values for each horizon
    """
    actuals_df = data[target_columns]
    forecasts_df = pd.concat(
        [df[target_col].iloc[-horizon_periods:] for df in outputs], axis=1
    )
    totals = actuals_df.sum()
    wmape_weights = np.array((totals / totals.sum()).values)

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

    for y_true, y_pred in zip(
        actuals_df.itertuples(index=False), forecasts_df.itertuples(index=False)
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

        metrics_df = metrics_df.append(metrics_row, ignore_index=True)

    metrics_df.set_index(data["ds"], inplace=True)

    return metrics_df


def _call_pandas_fsspec(pd_fn, filename, storage_options, **kwargs):
    if fsspec.utils.get_protocol(filename) == "file":
        return pd_fn(filename, **kwargs)
    return pd_fn(filename, storage_options=storage_options, **kwargs)


def _load_data(filename, format, storage_options, columns, **kwargs):
    if not format:
        _, format = os.path.splitext(filename)
        format = format[1:]
    if format in ["json", "clipboard", "excel", "csv", "feather", "hdf"]:
        read_fn = getattr(pd, f"read_{format}")
        data = _call_pandas_fsspec(read_fn, filename, storage_options=storage_options)
        if columns:
            # keep only these columns, done after load because only CSV supports stream filtering
            data = data[columns]
        return data
    raise ForecastInputDataError(f"Unrecognized format: {format}")


def _write_data(data, filename, format, storage_options, index=False, **kwargs):
    if not format:
        _, format = os.path.splitext(filename)
        format = format[1:]
    if format in ["json", "clipboard", "excel", "csv", "feather", "hdf"]:
        write_fn = getattr(data, f"to_{format}")
        return _call_pandas_fsspec(
            write_fn, filename, index=index, storage_options=storage_options
        )
    raise ForecastInputDataError(f"Unrecognized format: {format}")


def _merge_category_columns(data, target_category_columns):
    return data.apply(
        lambda x: "__".join([str(x[col]) for col in target_category_columns]), axis=1
    )


def _clean_data(data, target_column, datetime_column, target_category_columns=None):
    # Todo: KNN Imputer?
    if target_category_columns is not None:
        data["__Series__"] = _merge_category_columns(data, target_category_columns)
        unique_categories = data["__Series__"].unique()

        df = pd.DataFrame()
        new_target_columns = []

        for cat in unique_categories:
            data_cat = data[data["__Series__"] == cat].rename(
                {target_column: f"{target_column}_{cat}"}, axis=1
            )
            data_cat_clean = data_cat.drop("__Series__", axis=1).set_index(
                datetime_column
            )
            df = pd.concat([df, data_cat_clean], axis=1)
            new_target_columns.append(f"{target_column}_{cat}")
        df = df.reset_index()

        return df.fillna(0), new_target_columns

    raise ForecastSchemaYamlError(
        f"Either target_columns, target_category_columns, or datetime_column not specified."
    )


def _validate_and_clean_data(
    cat: str, horizon: int, primary: pd.DataFrame, additional: pd.DataFrame
):
    """
    Checks compatibility between primary and additional dataframe for a category.

    Parameters
    ----------
        cat: (str)
         Category for which data is being validated.
        horizon: (int)
         horizon value for the forecast.
        primary: (pd.DataFrame)
         primary dataframe.
        additional: (pd.DataFrame)
         additional dataframe.

    Returns
    -------
        (pd.DataFrame, pd.DataFrame) or (None, None)
         Updated primary and additional dataframe or None values if the validation criteria does not satisfy.
    """
    # Additional data should have future values for horizon
    data_row_count = primary.shape[0]
    data_add_row_count = additional.shape[0]
    additional_surplus = data_add_row_count - horizon - data_row_count
    if additional_surplus < 0:
        logger.warn(
            "Forecast for {} will not be generated since additional data has less values({}) than"
            " horizon({}) + primary data({})".format(
                cat, data_add_row_count, horizon, data_row_count
            )
        )
        return None, None
    elif additional_surplus > 0:
        # Removing surplus future data in additional
        additional.drop(additional.tail(additional_surplus).index, inplace=True)

    # Dates in primary data should be subset of additional data
    dates_in_data = primary.index.tolist()
    dates_in_additional = additional.index.tolist()
    if not set(dates_in_data).issubset(set(dates_in_additional)):
        logger.warn(
            "Forecast for {} will not be generated since the dates in primary and additional do not"
            " match".format(cat)
        )
        return None, None
    return primary, additional


def _build_indexed_datasets(
    data,
    target_column,
    datetime_column,
    horizon,
    target_category_columns=None,
    additional_data=None,
    metadata_data=None,
):
    df_by_target = dict()
    categories = []

    if target_category_columns is None:
        if additional_data is None:
            df_by_target[target_column] = data.fillna(0)
        else:
            df_by_target[target_column] = pd.concat(
                [
                    data.set_index(datetime_column).fillna(0),
                    additional_data.set_index(datetime_column).fillna(0),
                ],
                axis=1,
            ).reset_index()
        return df_by_target, target_column, categories

    data["__Series__"] = _merge_category_columns(data, target_category_columns)
    unique_categories = data["__Series__"].unique()
    invalid_categories = []
    for cat in unique_categories:
        data_by_cat = data[data["__Series__"] == cat].rename(
            {target_column: f"{target_column}_{cat}"}, axis=1
        )
        data_by_cat_clean = (
            data_by_cat.drop(target_category_columns + ["__Series__"], axis=1)
            .set_index(datetime_column)
            .fillna(0)
        )
        if additional_data is not None:
            additional_data["__Series__"] = _merge_category_columns(
                additional_data, target_category_columns
            )
            data_add_by_cat = additional_data[
                additional_data["__Series__"] == cat
            ].rename({target_column: f"{target_column}_{cat}"}, axis=1)
            data_add_by_cat_clean = (
                data_add_by_cat.drop(target_category_columns + ["__Series__"], axis=1)
                .set_index(datetime_column)
                .fillna(0)
            )

            valid_primary, valid_add = _validate_and_clean_data(
                cat, horizon, data_by_cat_clean, data_add_by_cat_clean
            )
            if valid_primary is None:
                invalid_categories.append(cat)
                data_by_cat_clean = None
            else:
                data_by_cat_clean = pd.concat([valid_add, valid_primary], axis=1)
        if data_by_cat_clean is not None:
            df_by_target[f"{target_column}_{cat}"] = data_by_cat_clean.reset_index()

    new_target_columns = list(df_by_target.keys())
    remaining_categories = set(unique_categories) - set(invalid_categories)
    if not len(remaining_categories):
        raise ForecastInputDataError(
            "Stopping forecast operator as there is no data that meets the validation criteria."
        )
    return df_by_target, new_target_columns, remaining_categories


def _build_metrics_df(y_true, y_pred, column_name):
    metrics = dict()
    metrics["sMAPE"] = smape(actual=y_true, predicted=y_pred)
    metrics["MAPE"] = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
    metrics["RMSE"] = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))
    metrics["r2"] = r2_score(y_true=y_true, y_pred=y_pred)
    metrics["Explained Variance"] = explained_variance_score(
        y_true=y_true, y_pred=y_pred
    )
    return pd.DataFrame.from_dict(metrics, orient="index", columns=[column_name])


def evaluate_metrics(target_columns, data, outputs, target_col="yhat"):
    total_metrics = pd.DataFrame()
    for idx, col in enumerate(target_columns):
        try:
            y_true = np.asarray(data[col])
            y_pred = np.asarray(outputs[idx][target_col][: len(y_true)])

            metrics_df = _build_metrics_df(
                y_true=y_true, y_pred=y_pred, column_name=col
            )
            total_metrics = pd.concat([total_metrics, metrics_df], axis=1)
        except:
            logger.warn(f"Failed to generate training metrics for target_series: {col}")
    return total_metrics


def _select_plot_list(fn, target_columns):
    import datapane as dp

    blocks = [dp.Plot(fn(i, col), label=col) for i, col in enumerate(target_columns)]
    return dp.Select(blocks=blocks) if len(target_columns) > 1 else blocks[0]


def _add_unit(num, unit):
    return f"{num} {unit}"


def get_forecast_plots(
    data,
    outputs,
    target_columns,
    test_data=None,
    ds_col=None,
    ds_forecast_col=None,
    forecast_col_name="yhat",
    ci_col_names=None,
    ci_interval_width=0.95,
):
    if ds_forecast_col is None:
        ds_forecast_col = ds_col

    def plot_forecast_plotly(idx, col):
        fig = go.Figure()
        if (
            (ci_col_names is not None)
            and (ci_col_names[0] in outputs[idx].columns)
            and (ci_col_names[1] in outputs[idx].columns)
        ):
            fig.add_traces(
                [
                    go.Scatter(
                        x=ds_forecast_col,
                        y=outputs[idx][ci_col_names[0]],
                        mode="lines",
                        line_color="rgba(0,0,0,0)",
                        showlegend=False,
                    ),
                    go.Scatter(
                        x=ds_forecast_col,
                        y=outputs[idx][ci_col_names[1]],
                        mode="lines",
                        line_color="rgba(0,0,0,0)",
                        name=f"{ci_interval_width*100}% confidence interval",
                        fill="tonexty",
                        fillcolor="rgba(211, 211, 211, 0.5)",
                    ),
                ]
            )
        if test_data is not None and col in test_data:
            fig.add_trace(
                go.Scatter(
                    x=test_data["ds"],
                    y=test_data[col],
                    mode="markers",
                    marker_color="green",
                    name="Actual",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=ds_col,
                y=data[col],
                mode="markers",
                marker_color="black",
                name="Historical",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ds_forecast_col,
                y=outputs[idx][forecast_col_name],
                mode="lines+markers",
                line_color="blue",
                name="Forecast",
            )
        )
        fig.add_vline(
            x=ds_col[-1:].values[0], line_width=1, line_dash="dash", line_color="gray"
        )
        return fig

    return _select_plot_list(plot_forecast_plotly, target_columns)


def human_time_friendly(seconds):
    TIME_DURATION_UNITS = (
        ("week", 60 * 60 * 24 * 7),
        ("day", 60 * 60 * 24),
        ("hour", 60 * 60),
        ("min", 60),
    )
    if seconds == 0:
        return "inf"
    accumulator = []
    for unit, div in TIME_DURATION_UNITS:
        amount, seconds = divmod(float(seconds), div)
        if amount > 0:
            accumulator.append(
                "{} {}{}".format(int(amount), unit, "" if amount == 1 else "s")
            )
    accumulator.append("{} secs".format(round(seconds, 2)))
    return ", ".join(accumulator)


def select_auto_model(columns: List[str]) -> str:
    """
    Selects AutoMLX or Arima model based on column count.

    If the number of columns is less than or equal to the maximum allowed for AutoMLX,
    returns 'AutoMLX'. Otherwise, returns 'Arima'.

    Parameters
    ------------
    columns:  List
            The list of columns.

    Returns
    --------
    str
        The type of the model.
    """
    if columns != None and len(columns) > MAX_COLUMNS_AUTOMLX:
        return SupportedModels.Arima
    return SupportedModels.AutoMLX


class GradioLLM:
    def __init__(self, url: str) -> None:
        from gradio_client import Client

        logging.basicConfig(level=logging.DEBUG)

        self.client = Client(url, serialize=True)

    def complete(self, completion: str, max_length=256) -> str:
        prompt = [[completion, ""]]

        with tempfile.NamedTemporaryFile(mode="w") as temp:
            json.dump(prompt, temp)
            temp.flush()

            results_file = self.client.predict(
                temp.name,
                max_length,  # int | float (numeric value between 256 and 4096)
                0.2,  # int | float (numeric value between 0.2 and 2.0)
                0.1,  # int | float (numeric value between 0.1 and 1.0)
                fn_index=2,
            )

            with open(results_file) as t:
                try:
                    data = t.read()
                    return json.loads(data)
                except Exception as ex:
                    logging.error(ex)
                    return None


def describe_metrics(llm_endpoint: str, metrics_str: str, algorithm_name: str):
    """
    Formats the metrics string into a query and submits it to LLM.

    Returns the formatted LLM response

    Parameters
    ------------
    columns:  str
        The ip address of the llm that can be invoked from the operator
    metrics_str: str
        A string version of the metrics being described

    Returns
    --------
    str
        The formatted text of the LLM response
    """

    BASE_METRICS_PROMPT = f"""The following table summarises the evaluation metrics for a machine learning forecasting model.
The columns of the table are the targets the model is forecasting and the rows show the performance across each metric.
Please evaluate the performance of the model across each target and then summarise the overall
strength of the model. The algorithm used was {algorithm_name}, add commentary on how this model type works.

Metrics:

"""
    llm = GradioLLM(llm_endpoint)
    metric_prompt = BASE_METRICS_PROMPT + metrics_str
    logger.debug(f"The full metrics prompt is: {metric_prompt}")
    result = llm.complete(metric_prompt, max_length=512)
    logger.debug(f"The LLM-generated response is: {result[0][1]}")
    return str(result[0][1])
