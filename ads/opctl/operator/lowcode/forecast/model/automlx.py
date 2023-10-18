#!/usr/bin/env python
# -*- coding: utf-8 -*--
import traceback

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd
import numpy as np
from ads.common.decorator.runtime_dependency import runtime_dependency
from ads.opctl.operator.lowcode.forecast.const import AUTOMLX_METRIC_MAP
from sktime.forecasting.model_selection import temporal_train_test_split
from ads.opctl import logger

from .. import utils
from .base_model import ForecastOperatorBaseModel
from ..operator_config import ForecastOperatorConfig


AUTOMLX_N_ALGOS_TUNED = 4
AUTOMLX_DEFAULT_SCORE_METRIC = "neg_sym_mean_abs_percent_error"


# TODO: ODSC-44785 Fix the error message, before GA.
class AutoMLXOperatorModel(ForecastOperatorBaseModel):
    """Class representing AutoMLX operator model."""

    def __init__(self, config: ForecastOperatorConfig):
        super().__init__(config)
        self.global_explanation = {}
        self.local_explanation = {}

    @runtime_dependency(
        module="automl",
        err_msg=(
            "Please run `pip3 install oracle-automlx==23.2.3` to install the required dependencies for automlx."
            "Please run `pip3 install oracle-automlx==23.2.3` to install the required dependencies for automlx."
        ),
    )
    def _build_model(self) -> pd.DataFrame:
        from automl import init

        init(engine="local", check_deprecation_warnings=False)

        full_data_dict = self.full_data_dict

        models = dict()
        outputs = dict()
        outputs_legacy = []
        selected_models = dict()
        date_column = self.spec.datetime_column.name
        horizon = self.spec.horizon.periods

        # Clean up kwargs for pass through
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
        ] = self.spec.preprocessing or model_kwargs_cleaned.get("preprocessing", True)

        for i, (target, df) in enumerate(full_data_dict.items()):
            logger.info("Running automl for {} at position {}".format(target, i))
            series_values = df[df[target].notna()]
            # drop NaNs for the time period where data wasn't recorded
            series_values.dropna(inplace=True)
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.set_index(date_column)
            if len(df.columns) > 1:
                # when additional columns are present
                y_train, y_test = temporal_train_test_split(df, test_size=horizon)
                forecast_x = y_test.drop(target, axis=1)
            else:
                y_train = df
                forecast_x = None
            logger.info(
                "Time Index is",
                "" if y_train.index.is_monotonic else "NOT",
                "monotonic.",
            )
            model = automl.Pipeline(
                task="forecasting",
                **model_kwargs_cleaned,
            )
            model.fit(
                X=y_train.drop(target, axis=1),
                y=pd.DataFrame(y_train[target]),
                time_budget=time_budget,
            )
            logger.info("Selected model: {}".format(model.selected_model_))
            logger.info(
                "Selected model params: {}".format(model.selected_model_params_)
            )
            summary_frame = model.forecast(
                X=forecast_x,
                periods=horizon,
                alpha=1 - ((self.spec.confidence_interval_width or 0.5) / 100),
            )
            # Collect Outputs
            selected_models[target] = {
                "series_id": target,
                "selected_model": model.selected_model_,
                "model_params": model.selected_model_params_,
            }
            models[target] = model
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
            outputs[target] = summary_frame
            outputs_legacy.append(summary_frame)

        logger.info("===========Forecast Generated===========")
        outputs_merged = pd.DataFrame()

        # Merge the outputs from each model into 1 df with all outputs by target and category
        col = self.original_target_column
        output_col = pd.DataFrame()
        yhat_lower_percentage = (
            100 - (self.spec.confidence_interval_width or 0.5) * 100
        ) // 2
        yhat_upper_name = "p" + str(int(100 - yhat_lower_percentage))
        yhat_lower_name = "p" + str(int(yhat_lower_percentage))
        for cat in self.categories:  # Note: add [:2] to restrict
            output_i = pd.DataFrame()
            output_i["Date"] = outputs[f"{col}_{cat}"]["ds"]
            output_i["Series"] = cat
            output_i["input_value"] = float("nan")
            output_i[f"fitted_value"] = float("nan")
            output_i[f"forecast_value"] = outputs[f"{col}_{cat}"]["yhat"]
            output_i[yhat_upper_name] = outputs[f"{col}_{cat}"]["yhat_upper"]
            output_i[yhat_lower_name] = outputs[f"{col}_{cat}"]["yhat_lower"]
            output_col = pd.concat([output_col, output_i])

        # output_col = output_col.sort_values(self.spec.datetime_column.name).reset_index(drop=True)
        output_col = output_col.reset_index(drop=True)
        outputs_merged = pd.concat([outputs_merged, output_col], axis=1)

        # Re-merge historical datas for processing
        data_merged = pd.concat(
            [v[v[k].notna()].set_index(date_column) for k, v in full_data_dict.items()],
            axis=1,
        ).reset_index()

        self.models = models
        self.outputs = outputs_legacy
        self.data = data_merged
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
        for i, (target, df) in enumerate(self.full_data_dict.items()):
            selected_models[target] = {
                "series_id": target,
                "selected_model": models[target].selected_model_,
                "model_params": models[target].selected_model_params_,
            }
        selected_models_df = pd.DataFrame(
            selected_models.items(), columns=["series_id", "best_selected_model"]
        )
        selected_df = selected_models_df["best_selected_model"].apply(pd.Series)
        selected_models_section = dp.Blocks(
            "### Best Selected model ", dp.Table(selected_df)
        )

        all_sections = [selected_models_text, selected_models_section]

        if self.spec.explain:
            # If the key is present, call the "explain_model" method
            self.explain_model()

            # Create a markdown text block for the global explanation section
            global_explanation_text = dp.Text(
                f"## Global Explanation of Models \n "
                "The following tables provide the feature attribution for the global explainability."
            )

            # Convert the global explanation data to a DataFrame
            global_explanation_df = pd.DataFrame(self.global_explanation)

            # Create a markdown section for the global explainability
            global_explanation_section = dp.Blocks(
                "### Global Explainability ",
                dp.Table(
                    global_explanation_df / global_explanation_df.sum(axis=0) * 100
                ),
                dp.Table(
                    global_explanation_df / global_explanation_df.sum(axis=0) * 100
                ),
            )

            local_explanation_text = dp.Text(f"## Local Explanation of Models \n ")
            blocks = [
                dp.Table(
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

        model_description = dp.Text(
            "The AutoMLx model automatically preprocesses, selects and engineers "
            "high-quality features in your dataset, which are then provided for further processing."
        )
        other_sections = all_sections
        forecast_col_name = "yhat"
        train_metrics = False
        ds_column_series = self.data[self.spec.datetime_column.name]
        ds_forecast_col = self.outputs[0]["ds"]
        ci_col_names = ["yhat_lower", "yhat_upper"]

        return (
            model_description,
            other_sections,
            forecast_col_name,
            train_metrics,
            ds_column_series,
            ds_forecast_col,
            ci_col_names,
        )

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
        temp = 0
        data_temp = pd.DataFrame(
            data,
            columns=[col for col in self.dataset_cols],
        )

        return self.models.get(self.series_id).forecast(
            X=data_temp, periods=data_temp.shape[0]
        )[self.series_id]

    @runtime_dependency(
        module="shap",
        err_msg=(
            "Please run `pip3 install shap` to install the required dependencies for model explanation."
            "Please run `pip3 install shap` to install the required dependencies for model explanation."
        ),
    )
    def explain_model(self) -> dict:
        """
        Generates an explanation for the model by using the SHAP (Shapley Additive exPlanations) library.
        This function calculates the SHAP values for each feature in the dataset and stores the results in the `global_explanation` dictionary.

        Returns
        -------
            dict: A dictionary containing the global explanation for each feature in the dataset.
                    The keys are the feature names and the values are the average absolute SHAP values.
        """
        from shap import KernelExplainer

        for series_id in self.target_columns:
            self.series_id = series_id
            self.dataset_cols = (
                self.full_data_dict.get(self.series_id)
                .set_index(self.spec.datetime_column.name)
                .drop(self.series_id, axis=1)
                .set_index(self.spec.datetime_column.name)
                .drop(self.series_id, axis=1)
                .columns
            )

            kernel_explnr = KernelExplainer(
                model=self._custom_predict_automlx,
                data=self.full_data_dict.get(self.series_id).set_index(
                    self.spec.datetime_column.name
                )[: -self.spec.horizon.periods][list(self.dataset_cols)],
            )

            kernel_explnr_vals = kernel_explnr.shap_values(
                self.full_data_dict.get(self.series_id).set_index(
                    self.spec.datetime_column.name
                )[: -self.spec.horizon.periods][list(self.dataset_cols)],
                nsamples=50,
            )

            print(kernel_explnr)
            self.global_explanation[self.series_id] = dict(
                zip(
                    self.dataset_cols,
                    np.average(np.absolute(kernel_explnr_vals), axis=0),
                )
            )

            self.local_explainer(kernel_explnr)

    def local_explainer(self, kernel_explainer) -> None:
        """
        Generate local explanations using a kernel explainer.

        Parameters
        ----------
            kernel_explainer: The kernel explainer object to use for generating explanations.
        """
        # Get the data for the series ID and select the relevant columns
        data = self.full_data_dict.get(self.series_id).set_index(
            self.spec.datetime_column.name
        )
        data = data[-self.spec.horizon.periods :][list(self.dataset_cols)]

        # Generate local SHAP values using the kernel explainer
        local_kernel_explnr_vals = kernel_explainer.shap_values(data, nsamples=50)

        # Convert the SHAP values into a DataFrame
        local_kernel_explnr_df = pd.DataFrame(
            local_kernel_explnr_vals, columns=self.dataset_cols
        )

        # set the index of the DataFrame to the datetime column
        local_kernel_explnr_df.index = data.index

        self.local_explanation[self.series_id] = local_kernel_explnr_df
