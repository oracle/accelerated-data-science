#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd
import numpy as np
import pmdarima as pm

from ads.opctl import logger

from .. import utils
from .base_model import ForecastOperatorBaseModel
from ..operator_config import ForecastOperatorConfig
import traceback


class ArimaOperatorModel(ForecastOperatorBaseModel):
    """Class representing ARIMA operator model."""

    def __init__(self, config: ForecastOperatorConfig):
        super().__init__(config)
        self.global_explanation = {}
        self.local_explanation = {}

    def _build_model(self) -> pd.DataFrame:
        full_data_dict = self.full_data_dict

        # Extract the Confidence Interval Width and convert to arima's equivalent - alpha
        if self.spec.confidence_interval_width is None:
            self.spec.confidence_interval_width = 1 - self.spec.model_kwargs.get(
                "alpha", 0.05
            )
        model_kwargs = self.spec.model_kwargs
        model_kwargs["alpha"] = 1 - self.spec.confidence_interval_width

        models = []
        outputs = dict()
        outputs_legacy = []

        for i, (target, df) in enumerate(full_data_dict.items()):
            # format the dataframe for this target. Dropping NA on target[df] will remove all future data
            le, df_encoded = utils._label_encode_dataframe(
                df, no_encode={self.spec.datetime_column.name, target}
            )

            df_encoded[self.spec.datetime_column.name] = pd.to_datetime(
                df_encoded[self.spec.datetime_column.name],
                format=self.spec.datetime_column.format,
            )
            df_clean = df_encoded.set_index(self.spec.datetime_column.name)
            data_i = df_clean[df_clean[target].notna()]

            # Assume that all columns passed in should be used as additional data
            additional_regressors = set(data_i.columns) - {
                target,
                self.spec.datetime_column.name,
            }
            logger.info(f"Additional Regressors Detected {list(additional_regressors)}")

            # Split data into X and y for arima tune method
            y = data_i[target]
            X_in = None
            if len(additional_regressors):
                X_in = data_i.drop(target, axis=1)

            # Build and fit model
            model = pm.auto_arima(y=y, X=X_in, **self.spec.model_kwargs)

            # Build future dataframe
            start_date = y.index.values[-1]
            n_periods = self.spec.horizon
            if len(additional_regressors):
                X = df_clean[df_clean[target].isnull()].drop(target, axis=1)
            else:
                X = pd.date_range(
                    start=start_date, periods=n_periods, freq=self.spec.freq
                )

            # Predict and format forecast
            yhat, conf_int = model.predict(
                n_periods=n_periods,
                X=X,
                return_conf_int=True,
                alpha=model_kwargs["alpha"],
            )
            yhat_clean = pd.DataFrame(yhat, index=yhat.index, columns=["yhat"])
            conf_int_clean = pd.DataFrame(
                conf_int, index=yhat.index, columns=["yhat_lower", "yhat_upper"]
            )
            forecast = pd.concat([yhat_clean, conf_int_clean], axis=1)
            logger.info(f"-----------------Model {i}----------------------")
            logger.info(forecast[["yhat", "yhat_lower", "yhat_upper"]].tail())

            # Collect all outputs
            models.append(model)
            outputs_legacy.append(
                forecast.reset_index().rename(columns={"index": "ds"})
            )
            outputs[target] = forecast

        self.models = models
        self.outputs = outputs_legacy

        logger.info("===========Done===========")
        outputs_merged = pd.DataFrame()

        # Merge the outputs from each model into 1 df with all outputs by target and category
        col = self.original_target_column
        output_col = pd.DataFrame()
        yhat_upper_percentage = int(100 - model_kwargs["alpha"] * 100 / 2)
        yhat_lower_name = "p" + str(int(100 - yhat_upper_percentage))
        yhat_upper_name = "p" + str(yhat_upper_percentage)
        for cat in self.categories:
            output_i = pd.DataFrame()

            output_i["Date"] = outputs[f"{col}_{cat}"].index
            output_i["Series"] = cat
            output_i["input_value"] = float("nan")
            output_i[f"fitted_value"] = float("nan")
            output_i[f"forecast_value"] = outputs[f"{col}_{cat}"]["yhat"].values
            output_i[yhat_upper_name] = outputs[f"{col}_{cat}"]["yhat_upper"].values
            output_i[yhat_lower_name] = outputs[f"{col}_{cat}"]["yhat_lower"].values
            output_col = pd.concat([output_col, output_i])

        # output_col = output_col.sort_values(operator.ds_column).reset_index(drop=True)
        output_col = output_col.reset_index(drop=True)
        outputs_merged = pd.concat([outputs_merged, output_col], axis=1)

        # Re-merge historical datas for processing
        data_merged = pd.concat(
            [
                v[v[k].notna()].set_index(self.spec.datetime_column.name)
                for k, v in full_data_dict.items()
            ],
            axis=1,
        ).reset_index()

        self.data = data_merged
        return outputs_merged

    def _generate_report(self):
        """The method that needs to be implemented on the particular model level."""
        import datapane as dp

        sec5_text = dp.Text(f"## ARIMA Model Parameters")
        blocks = [
            dp.HTML(m.summary().as_html(), label=self.target_columns[i])
            for i, m in enumerate(self.models)
        ]
        sec5 = dp.Select(blocks=blocks) if len(blocks) > 1 else blocks[0]
        all_sections = [sec5_text, sec5]

        if self.spec.generate_explanations:
            try:
                # If the key is present, call the "explain_model" method
                self.explain_model(
                    datetime_col_name=self.spec.datetime_column.name,
                    explain_predict_fn=self._custom_predict_arima,
                )

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
                    dp.Table(self.formatted_global_explanation),
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
            except Exception as e:
                logger.warn(f"Failed to generate Explanations with error: {e}.")
                logger.debug(f"Full Traceback: {traceback.format_exc()}")

        model_description = dp.Text(
            "An autoregressive integrated moving average, or ARIMA, is a statistical "
            "analysis model that uses time series data to either better understand the "
            "data set or to predict future trends. A statistical model is autoregressive if "
            "it predicts future values based on past values."
        )
        other_sections = all_sections
        ds_column_series = self.data[self.spec.datetime_column.name]
        ds_forecast_col = self.outputs[0].index
        ci_col_names = ["yhat_lower", "yhat_upper"]

        return (
            model_description,
            other_sections,
            ds_column_series,
            ds_forecast_col,
            ci_col_names,
        )

    def _custom_predict_arima(self, data):
        """
        Custom prediction function for ARIMA models.

        Parameters
        ----------
            data (array-like): The input data to be predicted.

        Returns
        -------
            array-like: The predicted values.

        """
        # Get the index of the current series id
        series_index = self.target_columns.index(self.series_id)

        # Use the ARIMA model to predict the values
        predictions = self.models[series_index].predict(X=data, n_periods=len(data))

        return predictions
