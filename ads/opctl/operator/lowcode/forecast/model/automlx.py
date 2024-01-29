#!/usr/bin/env python
# -*- coding: utf-8 -*--
import traceback

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd
import numpy as np
from ads.common.decorator.runtime_dependency import runtime_dependency
from ads.opctl.operator.lowcode.forecast.const import (
    AUTOMLX_METRIC_MAP,
    ForecastOutputColumns,
    SupportedModels,
)
from ads.opctl import logger

from ads.opctl.operator.lowcode.forecast.utils import convert_target
from .base_model import ForecastOperatorBaseModel
from ..operator_config import ForecastOperatorConfig
from .forecast_datasets import ForecastDatasets, ForecastOutput

AUTOMLX_N_ALGOS_TUNED = 4
AUTOMLX_DEFAULT_SCORE_METRIC = "neg_sym_mean_abs_percent_error"


class AutoMLXOperatorModel(ForecastOperatorBaseModel):
    """Class representing AutoMLX operator model."""

    def __init__(self, config: ForecastOperatorConfig, datasets: ForecastDatasets):
        super().__init__(config, datasets)
        self.global_explanation = {}
        self.local_explanation = {}
        self.train_metrics = True

    @runtime_dependency(
        module="automl",
        err_msg=(
            "Please run `pip3 install oracle-automlx==23.2.3` to install the required dependencies for automlx."
        ),
    )
    @runtime_dependency(
        module="sktime",
        err_msg=(
            "Please run `pip3 install sktime` to install the required dependencies for automlx."
        ),
    )
    def _build_model(self) -> pd.DataFrame:
        from automl import init
        from sktime.forecasting.model_selection import temporal_train_test_split

        init(
            engine="local",
            engine_opts={"n_jobs": -1, "model_n_jobs": -1},
            check_deprecation_warnings=False,
            logger=logger,
        )

        full_data_dict = self.datasets.get_data_by_series()

        models = dict()
        outputs = dict()
        outputs_legacy = dict()
        date_column = self.spec.datetime_column.name
        horizon = self.spec.horizon
        self.spec.confidence_interval_width = self.spec.confidence_interval_width or 0.8
        self.forecast_output = ForecastOutput(
            confidence_interval_width=self.spec.confidence_interval_width
        )

        # Clean up kwargs for pass through
        model_kwargs_cleaned = None

        if self.loaded_models is None:
            model_kwargs_cleaned = self.spec.model_kwargs.copy()
            model_kwargs_cleaned["n_algos_tuned"] = model_kwargs_cleaned.get(
                "n_algos_tuned", AUTOMLX_N_ALGOS_TUNED
            )
            model_kwargs_cleaned["score_metric"] = AUTOMLX_METRIC_MAP.get(
                self.spec.metric,
                model_kwargs_cleaned.get("score_metric", AUTOMLX_DEFAULT_SCORE_METRIC),
            )
            model_kwargs_cleaned.pop("task", None)
            time_budget = model_kwargs_cleaned.pop("time_budget", 0)
            model_kwargs_cleaned[
                "preprocessing"
            ] = self.spec.preprocessing or model_kwargs_cleaned.get(
                "preprocessing", True
            )

        for i, (s_id, df) in enumerate(full_data_dict.items()):
            try:
                target = self.original_target_column
                logger.debug("Running automl for {} at position {}".format(s_id, i))
                series_values = df[df[target].notna()]
                # drop NaNs for the time period where data wasn't recorded
                series_values.dropna(inplace=True)
                df[date_column] = pd.to_datetime(
                    df[date_column],
                    format=self.spec.datetime_column.format,  # TODO: could the format be different?
                )
                df = df.set_index(date_column)
                # if len(df.columns) > 1:
                # when additional columns are present
                y_train, y_test = temporal_train_test_split(df, test_size=horizon)
                forecast_x = y_test.drop(target, axis=1)
                # else:
                #     y_train = df
                #     forecast_x = None
                logger.debug(
                    "Time Index is" + ""
                    if y_train.index.is_monotonic
                    else "NOT" + "monotonic."
                )
                model = (
                    self.loaded_models[s_id] if self.loaded_models is not None else None
                )

                if model is None:
                    model = automl.Pipeline(
                        task="forecasting",
                        **model_kwargs_cleaned,
                    )
                    model.fit(
                        X=y_train.drop(target, axis=1),
                        y=pd.DataFrame(y_train[target]),
                        time_budget=time_budget,
                    )
                logger.debug("Selected model: {}".format(model.selected_model_))
                logger.debug(
                    "Selected model params: {}".format(model.selected_model_params_)
                )
                print(f"data: {forecast_x}, len(data): {horizon}")
                summary_frame = model.forecast(
                    X=forecast_x,
                    periods=horizon,
                    alpha=1 - (self.spec.confidence_interval_width / 100),
                )
                input_values = pd.Series(
                    y_train[target].values,
                    name="input_value",
                    index=y_train.index,
                )
                fitted_values_raw = model.predict(y_train.drop(target, axis=1))
                fitted_values = pd.Series(
                    fitted_values_raw[target].values,
                    name="fitted_value",
                    index=y_train.index,
                )

                summary_frame = pd.concat(
                    [input_values, fitted_values, summary_frame], axis=1
                )

                # Collect Outputs
                if self.loaded_models is None:
                    models[s_id] = model
                summary_frame = summary_frame.rename_axis("ds").reset_index()
                summary_frame = summary_frame.rename(
                    columns={
                        f"{target}_ci_upper": "yhat_upper",
                        f"{target}_ci_lower": "yhat_lower",
                        f"{target}": "yhat",
                    }
                )
                # In case of Naive model, model.forecast function call does not return confidence intervals.
                if "yhat_upper" not in summary_frame:
                    summary_frame["yhat_upper"] = np.NAN
                    summary_frame["yhat_lower"] = np.NAN
                outputs[s_id] = summary_frame

                self.model_parameters[s_id] = {
                    "framework": SupportedModels.AutoMLX,
                    "score_metric": model.score_metric,
                    "random_state": model.random_state,
                    "model_list": model.model_list,
                    "n_algos_tuned": model.n_algos_tuned,
                    "adaptive_sampling": model.adaptive_sampling,
                    "min_features": model.min_features,
                    "optimization": model.optimization,
                    "preprocessing": model.preprocessing,
                    "search_space": model.search_space,
                    "time_series_period": model.time_series_period,
                    "min_class_instances": model.min_class_instances,
                    "max_tuning_trials": model.max_tuning_trials,
                    "selected_model": model.selected_model_,
                    "selected_model_params": model.selected_model_params_,
                }
            except Exception as e:
                self.errors_dict[s_id] = {
                    "model_name": self.spec.model,
                    "error": str(e),
                }

        logger.debug("===========Forecast Generated===========")
        outputs_merged = pd.DataFrame()

        # Merge the outputs from each model into 1 df with all outputs by target and series
        yhat_upper_name = ForecastOutputColumns.UPPER_BOUND
        yhat_lower_name = ForecastOutputColumns.LOWER_BOUND
        for s_id in self.datasets.list_series_ids():
            output_i = pd.DataFrame()
            output_i["Date"] = outputs[s_id]["ds"]
            output_i["Series"] = s_id
            output_i["input_value"] = outputs[s_id]["input_value"]
            output_i[f"fitted_value"] = outputs[s_id]["fitted_value"]
            output_i[f"forecast_value"] = outputs[s_id]["yhat"]
            output_i[yhat_upper_name] = outputs[s_id]["yhat_upper"]
            output_i[yhat_lower_name] = outputs[s_id]["yhat_lower"]
            outputs_merged = pd.concat([outputs_merged, output_i])
            outputs_legacy[s_id] = output_i
            self.forecast_output.add_series_id(series_id=s_id, forecast=output_i)

        self.models = models if self.loaded_models is None else self.loaded_models

        return outputs_merged

    @runtime_dependency(
        module="datapane",
        err_msg=(
            "Please run `pip3 install datapane` to install the required dependencies for report generation."
        ),
    )
    def _generate_report(self):
        """
        Generate the report for the automlx model.

        Parameters
        ----------
        None

        Returns
        -------
            - model_description (datapane.Text): A Text component containing the description of the automlx model.
            - other_sections (List[Union[datapane.Text, datapane.Blocks]]): A list of Text and Blocks components representing various sections of the report.
            - forecast_col_name (str): The name of the forecasted column.
            - train_metrics (bool): A boolean value indicating whether to include train metrics in the report.
            - ds_column_series (pd.Series): The pd.Series object representing the datetime column of the dataset.
            - ds_forecast_col (pd.Series): The pd.Series object representing the forecasted column.
            - ci_col_names (List[str]): A list of column names for the confidence interval in the report.
        """
        import datapane as dp

        """The method that needs to be implemented on the particular model level."""
        selected_models_text = dp.Text(
            f"## Selected Models Overview \n "
            "The following tables provide information regarding the "
            "chosen model for each series and the corresponding parameters of the models."
        )
        selected_models = dict()
        models = self.models
        for i, (s_id, df) in enumerate(self.full_data_dict.items()):
            selected_models[s_id] = {
                "series_id": s_id,
                "selected_model": models[s_id].selected_model_,
                "model_params": models[s_id].selected_model_params_,
            }
        selected_models_df = pd.DataFrame(
            selected_models.items(), columns=["series_id", "best_selected_model"]
        )
        selected_df = selected_models_df["best_selected_model"].apply(pd.Series)
        selected_models_section = dp.Blocks(
            "### Best Selected Model", dp.DataTable(selected_df)
        )

        all_sections = [selected_models_text, selected_models_section]

        if self.spec.generate_explanations:
            try:
                # If the key is present, call the "explain_model" method
                self.explain_model()

                # Create a markdown text block for the global explanation section
                global_explanation_text = dp.Text(
                    f"## Global Explanation of Models \n "
                    "The following tables provide the feature attribution for the global explainability."
                )

                # Convert the global explanation data to a DataFrame
                global_explanation_df = pd.DataFrame(self.global_explanation)

                self.formatted_global_explanation = (
                    global_explanation_df / global_explanation_df.sum(axis=0) * 100
                )

                # Create a markdown section for the global explainability
                global_explanation_section = dp.Blocks(
                    "### Global Explainability ",
                    dp.DataTable(self.formatted_global_explanation),
                )

                aggregate_local_explanations = pd.DataFrame()
                for s_id, local_ex_df in self.local_explanation.items():
                    local_ex_df_copy = local_ex_df.copy()
                    local_ex_df_copy["Series"] = s_id
                    aggregate_local_explanations = pd.concat(
                        [aggregate_local_explanations, local_ex_df_copy], axis=0
                    )
                self.formatted_local_explanation = aggregate_local_explanations

                local_explanation_text = dp.Text(f"## Local Explanation of Models \n ")
                blocks = [
                    dp.DataTable(
                        local_ex_df.div(local_ex_df.abs().sum(axis=1), axis=0) * 100,
                        label=s_id,
                    )
                    for s_id, local_ex_df in self.local_explanation.items()
                ]
                local_explanation_section = (
                    dp.Select(blocks=blocks) if len(blocks) > 1 else blocks[0]
                )

                # Append the global explanation text and section to the "all_sections" list
                all_sections = all_sections + [
                    global_explanation_text,
                    global_explanation_section,
                    local_explanation_text,
                    local_explanation_section,
                ]
            except Exception as e:
                logger.warn(f"Failed to generate Explanations with error: {e}.")
                logger.debug(f"Full Traceback: {traceback.format_exc()}")

        model_description = dp.Text(
            "The AutoMLx model automatically preprocesses, selects and engineers "
            "high-quality features in your dataset, which are then provided for further processing."
        )
        other_sections = all_sections

        return (
            model_description,
            other_sections,
        )

    def get_explain_predict_fn(self, series_id):
        selected_model = self.models[series_id]

        def _custom_predict_fn(
            data,
            model=selected_model,
            dt_column_name=self.datasets._datetime_column_name,
            target_col=self.original_target_column,
        ):
            """
            data: ForecastDatasets.get_data_at_series(s_id)
            """
            data = data.drop(target_col, axis=1)
            data[dt_column_name] = pd.to_datetime(data[dt_column_name], unit="s")
            data = data.set_index(dt_column_name)
            print(f"data: {data}, len(data): {data.shape[0]}")
            return model.predict(X=data)[target_col]
            # forecast = pd.DataFrame()
            # for idx in range(data.shape[0]):
            #     forecast_i = model.forecast(X=data[idx:idx+1], periods=1)
            #     print(f"Looking at forecasts")
            #     print(type(forecast_i))
            #     print(forecast_i)
            #     forecast = pd.concat([forecast, forecast_i])
            # return forecast

        return _custom_predict_fn

    def _custom_predict_automlx(self, data):
        """
        Predicts the future values of a time series using the AutoMLX model.
        Parameters
        ----------
            data (numpy.ndarray): The input data to be used for prediction.

        Returns
        -------
            numpy.ndarray: The predicted future values of the time series.
        """
        data_temp = pd.DataFrame(
            data,
            columns=[col for col in self.dataset_cols],
        )

        return self.models.get(self.series_id).forecast(
            X=data_temp, periods=data_temp.shape[0]
        )[self.series_id]
