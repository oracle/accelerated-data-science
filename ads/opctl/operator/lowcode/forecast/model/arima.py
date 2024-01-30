#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd
import numpy as np
import pmdarima as pm
from joblib import Parallel, delayed

from ads.opctl import logger

from ads.opctl.operator.lowcode.forecast.utils import _label_encode_dataframe
from .base_model import ForecastOperatorBaseModel
from ..operator_config import ForecastOperatorConfig
import traceback
from .forecast_datasets import ForecastDatasets, ForecastOutput
from ..const import ForecastOutputColumns, SupportedModels


class ArimaOperatorModel(ForecastOperatorBaseModel):
    """Class representing ARIMA operator model."""

    def __init__(self, config: ForecastOperatorConfig, datasets: ForecastDatasets):
        super().__init__(config, datasets=datasets)
        self.global_explanation = {}
        self.local_explanation = {}
        self.train_metrics = True
        self.formatted_global_explanation = None
        self.formatted_local_explanation = None

    def _train_model(self, i, s_id, df):
        """Trains the ARIMA model for a given series of the dataset.

        Parameters
        ----------
        i: int
            The index of the series
        s_id: str
            The name of the series
        df: pd.DataFrame
            The dataframe containing the target data
        """
        try:
            # Extract the Confidence Interval Width and convert to arima's equivalent - alpha
            if self.spec.confidence_interval_width is None:
                self.spec.confidence_interval_width = 1 - self.spec.model_kwargs.get(
                    "alpha", 0.05
                )
            model_kwargs = self.spec.model_kwargs
            model_kwargs["alpha"] = 1 - self.spec.confidence_interval_width
            if "error_action" not in model_kwargs.keys():
                model_kwargs["error_action"] = "ignore"

            target = self.original_target_column

            # format the dataframe for this target. Dropping NA on target[df] will remove all future data
            le, df_encoded = _label_encode_dataframe(
                df, no_encode={self.spec.datetime_column.name, target}
            )

            df_encoded[self.spec.datetime_column.name] = pd.to_datetime(
                df_encoded[self.spec.datetime_column.name],
                format=self.spec.datetime_column.format,  # TODO: could the format be different?
            )
            df_clean = df_encoded.set_index(self.spec.datetime_column.name)
            data_i = df_clean[df_clean[target].notna()]

            # Assume that all columns passed in should be used as additional data
            additional_regressors = set(data_i.columns) - {
                target,
                self.spec.datetime_column.name,
            }
            logger.debug(
                f"Additional Regressors Detected {list(additional_regressors)}"
            )

            # Split data into X and y for arima tune method
            y = data_i[target]
            X_in = None
            if len(additional_regressors):
                X_in = data_i.drop(target, axis=1)

            model = self.loaded_models[s_id] if self.loaded_models is not None else None
            if model is None:
                # Build and fit model
                model = pm.auto_arima(y=y, X=X_in, **self.spec.model_kwargs)

            print(f"X_in: {X_in}, pred: {model.predict_in_sample(X=X_in)}")

            self.fitted_values[s_id] = pd.Series(
                model.predict_in_sample(X=X_in).values,
                index=data_i.index,
                name="fitted_values",
            )

            # Build future dataframe
            start_date = y.index.values[-1]
            n_periods = self.spec.horizon
            if len(additional_regressors):
                X = df_clean[df_clean[target].isnull()].drop(target, axis=1)
            else:
                X = pd.date_range(
                    start=start_date,
                    periods=n_periods,
                    freq=self.datasets.get_datetime_frequency(),
                )

            # Predict and format forecast
            yhat, conf_int = model.predict(
                n_periods=n_periods,
                X=X,
                return_conf_int=True,
                alpha=model_kwargs["alpha"],
            )
            yhat_clean = pd.DataFrame(yhat, index=yhat.index, columns=["yhat"])

            self.dt_columns[s_id] = df_encoded[self.spec.datetime_column.name]
            conf_int_clean = pd.DataFrame(
                conf_int, index=yhat.index, columns=["yhat_lower", "yhat_upper"]
            )
            forecast = pd.concat([yhat_clean, conf_int_clean], axis=1)
            logger.debug(f"-----------------Model {i}----------------------")
            logger.debug(forecast[["yhat", "yhat_lower", "yhat_upper"]].tail())

            # Collect all outputs
            # models.append(model)
            self.outputs_legacy.append(
                forecast.reset_index().rename(columns={"index": "ds"})
            )
            self.outputs[s_id] = forecast

            if self.loaded_models is None:
                self.models[s_id] = model

            params = vars(model).copy()
            for param in ["arima_res_", "endog_index_"]:
                if param in params:
                    params.pop(param)
            self.model_parameters[s_id] = {
                "framework": SupportedModels.Arima,
                **params,
            }

            logger.debug("===========Done===========")
        except Exception as e:
            self.errors_dict[s_id] = {"model_name": self.spec.model, "error": str(e)}

    def _build_model(self) -> pd.DataFrame:
        full_data_dict = self.datasets.get_data_by_series()

        self.forecast_output = ForecastOutput(
            confidence_interval_width=self.spec.confidence_interval_width
        )

        self.models = dict()
        self.outputs = dict()
        self.outputs_legacy = []
        self.fitted_values = dict()
        self.dt_columns = dict()

        Parallel(n_jobs=-1, require="sharedmem")(
            delayed(ArimaOperatorModel._train_model)(self, i, s_id, df)
            for self, (i, (s_id, df)) in zip(
                [self] * len(full_data_dict), enumerate(full_data_dict.items())
            )
        )

        if self.loaded_models is not None:
            self.models = self.loaded_models

        # Merge the outputs from each model into 1 df with all outputs by target and series
        output_col = pd.DataFrame()
        yhat_upper_name = ForecastOutputColumns.UPPER_BOUND
        yhat_lower_name = ForecastOutputColumns.LOWER_BOUND

        for s_id in self.datasets.list_series_ids():
            output_i = pd.DataFrame()
            output_i["Date"] = full_data_dict[s_id][self.spec.datetime_column.name]
            output_i["Series"] = s_id
            output_i = output_i.set_index("Date")

            output_i["input_value"] = full_data_dict[s_id][self.original_target_column]
            print(f"output_i: {output_i}, self.fitted_values: {self.fitted_values}")
            output_i["fitted_value"] = self.fitted_values[s_id]
            output_i["forecast_value"] = self.outputs[s_id]["yhat"]
            output_i[yhat_upper_name] = self.outputs[s_id]["yhat_upper"]
            output_i[yhat_lower_name] = self.outputs[s_id]["yhat_lower"]

            # output_i = output_i.reset_index(drop=False)
            output_col = pd.concat([output_col, output_i])
            self.forecast_output.add_series_id(series_id=s_id, forecast=output_i)

        output_col = output_col.reset_index(drop=True)

        return output_col

    def _generate_report(self):
        """The method that needs to be implemented on the particular model level."""
        import datapane as dp

        sec5_text = dp.Text(f"## ARIMA Model Parameters")
        blocks = [
            dp.HTML(
                m.summary().as_html(),
                label=s_id,
            )
            for i, (s_id, m) in enumerate(self.models.items())
        ]
        sec5 = dp.Select(blocks=blocks) if len(blocks) > 1 else blocks[0]
        all_sections = [sec5_text, sec5]

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
            "An autoregressive integrated moving average, or ARIMA, is a statistical "
            "analysis model that uses time series data to either better understand the "
            "data set or to predict future trends. A statistical model is autoregressive if "
            "it predicts future values based on past values."
        )
        other_sections = all_sections

        return (
            model_description,
            other_sections,
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
        date_col = self.spec.datetime_column.name
        data[date_col] = pd.to_datetime(data[date_col], unit="s")
        data = data.set_index(date_col)

        # Use the ARIMA model to predict the values
        predictions = self.models[self.series_id].predict(X=data, n_periods=len(data))

        return predictions

    def get_explain_predict_fn(self, series_id):
        selected_model = self.models[series_id]

        def _custom_predict_prophet(
            data,
            model=selected_model,
            dt_column_name=self.datasets._datetime_column_name,
            target_col=self.original_target_column,
        ):
            """
            data: ForecastDatasets.get_data_at_series(s_id)
            """
            data = data.drop([target_col], axis=1)
            data[dt_column_name] = pd.to_datetime(data[dt_column_name], unit="s")
            data = data.set_index(dt_column_name)
            return model.predict(X=data, n_periods=len(data))

        return _custom_predict_prophet
