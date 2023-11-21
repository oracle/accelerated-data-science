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

from .const import SupportedMetrics, SupportedModels
from .errors import ForecastInputDataError, ForecastSchemaYamlError
from .operator_config import ForecastOperatorSpec, ForecastOperatorConfig


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
    data: pd.DataFrame,
    output: pd.DataFrame,
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
    output: Pandas Dataframe
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
    """
    Assumptions:
    data and output have all the target columns.
    yhats in output are in the same order as in target_columns.
    Test data might not have sorted dates and the order of series also might differ.
    """

    # Select the data with correct order of target_columns.
    target_columns = list(set.intersection(set(target_columns), set(data.columns)))

    actuals_df = data[["ds"] + target_columns]

    # Concat the yhats in output and include only dates that are in test data
    forecasts_df = pd.DataFrame()
    for cat in output.list_categories():
        forecast_i = output.get_category(cat)[["Date", "forecast_value"]]
        forecast_i = forecast_i[forecast_i["Date"].isin(actuals_df["ds"])]
        forecasts_df = pd.concat([forecasts_df, forecast_i.set_index("Date")], axis=1)

    # Remove dates that are not there in output
    actuals_df = actuals_df[actuals_df["ds"].isin(forecasts_df.index.values)]

    if actuals_df.empty or forecasts_df.empty:
        return pd.DataFrame()

    totals = actuals_df.sum(numeric_only=True)
    wmape_weights = np.array((totals / totals.sum()).values)

    actuals_df = actuals_df.set_index("ds")

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


def _call_pandas_fsspec(pd_fn, filename, storage_options, **kwargs):
    if fsspec.utils.get_protocol(filename) == "file":
        return pd_fn(filename, **kwargs)
    elif fsspec.utils.get_protocol(filename) in ["http", "https"]:
        return pd_fn(filename, **kwargs)

    storage_options = storage_options or (
        default_signer() if ObjectStorageDetails.is_oci_path(filename) else {}
    )

    return pd_fn(filename, storage_options=storage_options, **kwargs)


def _load_data(filename, format, storage_options=None, columns=None, **kwargs):
    if not format:
        _, format = os.path.splitext(filename)
        format = format[1:]
    if format in ["json", "clipboard", "excel", "csv", "feather", "hdf"]:
        read_fn = getattr(pd, f"read_{format}")
        data = _call_pandas_fsspec(read_fn, filename, storage_options=storage_options)
    elif format in ["tsv"]:
        data = _call_pandas_fsspec(
            pd.read_csv, filename, storage_options=storage_options, sep="\t"
        )
    else:
        raise ForecastInputDataError(f"Unrecognized format: {format}")
    if columns:
        # keep only these columns, done after load because only CSV supports stream filtering
        data = data[columns]
    return data


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
    result = data.apply(
        lambda x: "__".join([str(x[col]) for col in target_category_columns]), axis=1
    )
    return result if not result.empty else pd.Series([], dtype=str)


def _clean_data(data, target_column, datetime_column, target_category_columns=None):
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
            "Forecast for {} will not be generated since additional data has fewer values({}) than"
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

    if additional_data is not None and target_column in additional_data.columns:
        logger.warn(f"Dropping column '{target_column}' from additional_data")
        additional_data.drop(target_column, axis=1, inplace=True)
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
    try:
        metrics["r2"] = linregress(y_true, y_pred).rvalue ** 2
    except:
        metrics["r2"] = r2_score(y_true=y_true, y_pred=y_pred)
    metrics["Explained Variance"] = explained_variance_score(
        y_true=y_true, y_pred=y_pred
    )
    return pd.DataFrame.from_dict(metrics, orient="index", columns=[column_name])


def evaluate_train_metrics(
    target_columns, datasets, output, datetime_col, target_col="yhat"
):
    """
    Training metrics
    """
    total_metrics = pd.DataFrame()
    for idx, col in enumerate(target_columns):
        try:
            forecast_by_col = output.get_target_category(col)[
                ["input_value", "Date", "fitted_value"]
            ].dropna()
            y_true = forecast_by_col["input_value"].values
            y_pred = forecast_by_col["fitted_value"].values
            metrics_df = _build_metrics_df(
                y_true=y_true, y_pred=y_pred, column_name=col
            )
            total_metrics = pd.concat([total_metrics, metrics_df], axis=1)
        except Exception as e:
            logger.warn(f"Failed to generate training metrics for target_series: {col}")
            logger.debug(f"Recieved Error Statement: {e}")
    return total_metrics


def _select_plot_list(fn, target_columns):
    import datapane as dp

    blocks = [dp.Plot(fn(i, col), label=col) for i, col in enumerate(target_columns)]
    return dp.Select(blocks=blocks) if len(target_columns) > 1 else blocks[0]


def _add_unit(num, unit):
    return f"{num} {unit}"


def get_forecast_plots(
    forecast_output,
    target_columns,
    horizon,
    test_data=None,
    ci_interval_width=0.95,
):
    def plot_forecast_plotly(idx, col):
        fig = go.Figure()
        forecast_i = forecast_output.get_target_category(col)
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
    date_column = operator_config.spec.datetime_column.name
    datetimes = pd.to_datetime(
        datasets.original_user_data[date_column].drop_duplicates()
    )
    freq_in_secs = datetimes.tail().diff().min().total_seconds()
    if datasets.original_additional_data is not None:
        num_of_additional_cols = len(datasets.original_additional_data.columns) - 2
    else:
        num_of_additional_cols = 0
    row_count = len(datasets.original_user_data.index)
    number_of_series = len(datasets.categories)
    if (
        num_of_additional_cols < 15
        and row_count < 10000
        and number_of_series < 10
        and freq_in_secs > 3600
    ):
        return SupportedModels.AutoMLX
    elif row_count < 10000 and number_of_series > 10:
        operator_config.spec.model_kwargs["model_list"] = "fast_parallel"
        return SupportedModels.AutoTS
    elif row_count < 20000 and number_of_series > 10:
        operator_config.spec.model_kwargs["model_list"] = "superfast"
        return SupportedModels.AutoTS
    elif row_count > 20000:
        return SupportedModels.NeuralProphet
    else:
        return SupportedModels.NeuralProphet


def get_frequency_of_datetime(data: pd.DataFrame, dataset_info: ForecastOperatorSpec):
    """
    Function checks if the data is compatible with the model selected

    Parameters
    ------------
    data:  pd.DataFrame
            primary dataset
    dataset_info:  ForecastOperatorSpec

    Returns
    --------
    None

    """
    date_column = dataset_info.datetime_column.name
    datetimes = pd.to_datetime(
        data[date_column].drop_duplicates(), format=dataset_info.datetime_column.format
    )
    freq = pd.DatetimeIndex(datetimes).inferred_freq
    if dataset_info.model == SupportedModels.AutoMLX:
        freq_in_secs = datetimes.tail().diff().min().total_seconds()
        if abs(freq_in_secs) < 3600:
            message = (
                "{} requires data with a frequency of at least one hour. Please try using a different model,"
                " or select the 'auto' option.".format(SupportedModels.AutoMLX, freq)
            )
            raise Exception(message)
    return freq


def default_signer(**kwargs):
    os.environ["EXTRA_USER_AGENT_INFO"] = "Forecast-Operator"
    from ads.common.auth import default_signer

    return default_signer(**kwargs)


# Disable
def block_print():
    sys.stdout = open(os.devnull, "w")


# Restore
def enable_print():
    sys.stdout = sys.__stdout__
