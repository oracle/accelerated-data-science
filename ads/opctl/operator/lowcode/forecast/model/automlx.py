#!/usr/bin/env python
# -*- coding: utf-8 -*--
import traceback

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import datapane as dp
import pandas as pd
import numpy as np
from ads.common.decorator.runtime_dependency import runtime_dependency
from ads.opctl.operator.lowcode.forecast.const import automlx_metric_dict
from sktime.forecasting.model_selection import temporal_train_test_split
from ads.opctl import logger

from .. import utils
from .base_model import ForecastOperatorBaseModel
from ..operator_config import ForecastOperatorConfig

# breakpoint()


# TODO: ODSC-44785 Fix the error message, before GA.
class AutoMLXOperatorModel(ForecastOperatorBaseModel):
    """Class representing AutoMLX operator model."""

    def __init__(self, config: ForecastOperatorConfig):
        super().__init__(config)
        self.global_explanation = {}

    @runtime_dependency(
        module="automl",
        err_msg=(
            "Please run `pip3 install "
            "--extra-index-url=https://artifacthub-phx.oci.oraclecorp.com/artifactory/api/pypi/automlx-pypi/simple/automlx==23.2.1` "
            "to install the required dependencies for automlx."
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
        n_algos_tuned = self.spec.model_kwargs.get("n_algos_tuned", 4)
        date_column = self.spec.datetime_column.name
        horizon = self.spec.horizon.periods
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
                vikas,
            )

            model = automl.Pipeline(
                task="forecasting",
                n_algos_tuned=n_algos_tuned,
                score_metric=automlx_metric_dict[self.spec.metric],
            )

            model.fit(X=y_train.drop(target, axis=1), y=pd.DataFrame(y_train[target]))
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
        for cat in self.categories:  # Note: add [:2] to restrict
            output_i = pd.DataFrame()
            output_i[self.spec.datetime_column.name] = outputs[f"{col}_{cat}"]["ds"]
            output_i["Series"] = cat
            output_i[f"{col}_forecast"] = outputs[f"{col}_{cat}"]["yhat"]
            output_i[f"{col}_forecast_upper"] = outputs[f"{col}_{cat}"]["yhat_upper"]
            output_i[f"{col}_forecast_lower"] = outputs[f"{col}_{cat}"]["yhat_lower"]
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

    def _generate_report(self):
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

        model_description = dp.Text(
            "The automlx model automatically preprocesses, selects and engineers "
            "high-quality features in your dataset, which then given to an automatically "
            "chosen and optimized machine learning model.."
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
        temp = 0
        data_temp = pd.DataFrame(
            data,  # [:, :len(self.dataset_cols)],
            columns=[col for col in self.dataset_cols],
        )
        # if data.shape[0] == 1:
        #     orig_data_index = self.full_data_dict.get(self.series_id)[:-4].set_index(
        #         list(self.dataset_cols)).index
        #     new_data_index = data_temp.set_index(list(self.dataset_cols)).index
        #     prediction_index = self.full_data_dict.get(self.series_id)[:-4][
        #         orig_data_index.isin(new_data_index)].index.values

        return self.models.get(self.series_id).forecast(
            X=data_temp.drop(self.series_id, axis=1), periods=data_temp.shape[0]
        )[self.series_id]

    def explain_model(self) -> dict:
        """
        explain the automlx model using local and global explanations
        """
        try:
            from shap import KernelExplainer
        except Exception as ex:
            print(
                "Please run `pip install shap to install "
                "the required dependencies for ADS CLI."
            )
            logger.debug(ex)
            logger.debug(traceback.format_exc())
            exit()
        for series_id in self.target_columns:
            self.series_id = series_id
            self.dataset_cols = (
                self.full_data_dict.get(self.series_id)
                .set_index(self.spec.datetime_column.name)
                .columns
            )

            # if not self.models.get(self.series_id).selected_model_params_.get("use_X", False):
            #     self.dataset_cols = {self.series_id}

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
