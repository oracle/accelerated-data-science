#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd
import numpy as np
import pmdarima as pm
from joblib import Parallel, delayed

from ads.opctl import logger

from ads.opctl.operator.lowcode.forecast.utils import _label_encode_dataframe
from ads.opctl.operator.lowcode.common.utils import seconds_to_datetime
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
        self.formatted_global_explanation = None
        self.formatted_local_explanation = None
        self.constant_cols = {}

    def set_kwargs(self):
        # Extract the Confidence Interval Width and convert to arima's equivalent - alpha
        if self.spec.confidence_interval_width is None:
            self.spec.confidence_interval_width = 1 - self.spec.model_kwargs.get(
                "alpha", 0.05
            )
        model_kwargs = self.spec.model_kwargs
        model_kwargs["alpha"] = 1 - self.spec.confidence_interval_width
        if "error_action" not in model_kwargs.keys():
            model_kwargs["error_action"] = "ignore"
        return model_kwargs

    def preprocess(self, data, series_id):  # TODO: re-use self.le for explanations
        self.le[series_id], df_encoded = _label_encode_dataframe(
            data,
            no_encode={self.spec.datetime_column.name, self.original_target_column},
        )
        return df_encoded.set_index(self.spec.datetime_column.name)

    def _train_model(self, i, s_id, df, model_kwargs):
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
            target = self.original_target_column
            self.forecast_output.init_series_output(series_id=s_id, data_at_series=df)
            # If trend is constant, remove constant columns
            if "trend" not in model_kwargs or model_kwargs["trend"] == "c":
                self.constant_cols[s_id] = df.columns[df.nunique() == 1]
                df = df.drop(columns=self.constant_cols[s_id])

            # format the dataframe for this target. Dropping NA on target[df] will remove all future data
            data = self.preprocess(df, s_id)
            data_i = self.drop_horizon(data)

            # Split data into X and y for arima tune method
            y = data_i[target]
            X_in = data_i.drop(target, axis=1) if len(data_i.columns) > 1 else None
            X_pred = self.get_horizon(data).drop(target, axis=1)

            if self.loaded_models is not None and s_id in self.loaded_models:
                model = self.loaded_models[s_id]
            else:
                # Build and fit model
                model = pm.auto_arima(y=y, X=X_in, **model_kwargs)

            fitted_values = model.predict_in_sample(X=X_in).values

            # Predict and format forecast
            yhat, conf_int = model.predict(
                n_periods=self.spec.horizon,
                X=X_pred,
                return_conf_int=True,
                alpha=model_kwargs["alpha"],
            )
            yhat_clean = pd.DataFrame(yhat, index=yhat.index, columns=["yhat"])

            conf_int_clean = pd.DataFrame(
                conf_int, index=yhat.index, columns=["yhat_lower", "yhat_upper"]
            )
            forecast = pd.concat([yhat_clean, conf_int_clean], axis=1)
            logger.debug(f"-----------------Model {i}----------------------")
            logger.debug(forecast[["yhat", "yhat_lower", "yhat_upper"]].tail())

            self.forecast_output.populate_series_output(
                series_id=s_id,
                fit_val=fitted_values,
                forecast_val=self.get_horizon(forecast["yhat"]).values,
                upper_bound=self.get_horizon(forecast["yhat_upper"]).values,
                lower_bound=self.get_horizon(forecast["yhat_lower"]).values,
            )

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
            logger.debug(f"Encountered Error: {e}. Skipping.")

    def _build_model(self) -> pd.DataFrame:
        full_data_dict = self.datasets.get_data_by_series()
        self.models = dict()
        self.additional_regressors = self.datasets.get_additional_data_column_names()
        model_kwargs = self.set_kwargs()
        self.forecast_output = ForecastOutput(
            confidence_interval_width=self.spec.confidence_interval_width,
            horizon=self.spec.horizon,
            target_column=self.original_target_column,
            dt_column=self.spec.datetime_column.name,
        )

        Parallel(n_jobs=-1, require="sharedmem")(
            delayed(self._train_model)(i, s_id, df, model_kwargs.copy())
            for (i, (s_id, df)) in enumerate(full_data_dict.items())
        )

        return self.forecast_output.get_forecast_long()

    def _generate_report(self):
        """The method that needs to be implemented on the particular model level."""
        import report_creator as rc

        all_sections = []
        if len(self.models) > 0:
            sec5_text = rc.Heading("ARIMA Model Parameters", level=2)
            blocks = [
                rc.Html(
                    m.summary().as_html(),
                    label=s_id,
                )
                for i, (s_id, m) in enumerate(self.models.items())
            ]
            sec5 = rc.Select(blocks=blocks)
            all_sections = [sec5_text, sec5]

        if self.spec.generate_explanations:
            try:
                # If the key is present, call the "explain_model" method
                self.explain_model()

                # Convert the global explanation data to a DataFrame
                global_explanation_df = pd.DataFrame(self.global_explanation)

                self.formatted_global_explanation = (
                    global_explanation_df / global_explanation_df.sum(axis=0) * 100
                )
                self.formatted_global_explanation = (
                    self.formatted_global_explanation.rename(
                        {self.spec.datetime_column.name: ForecastOutputColumns.DATE},
                        axis=1,
                    )
                )

                # Create a markdown section for the global explainability
                global_explanation_section = rc.Block(
                    rc.Heading("Global Explanation of Models", level=2),
                    rc.Text(
                        "The following tables provide the feature attribution for the global explainability."
                    ),
                    rc.DataTable(self.formatted_global_explanation, index=True),
                )

                aggregate_local_explanations = pd.DataFrame()
                for s_id, local_ex_df in self.local_explanation.items():
                    local_ex_df_copy = local_ex_df.copy()
                    local_ex_df_copy["Series"] = s_id
                    aggregate_local_explanations = pd.concat(
                        [aggregate_local_explanations, local_ex_df_copy], axis=0
                    )
                self.formatted_local_explanation = aggregate_local_explanations

                blocks = [
                    rc.DataTable(
                        local_ex_df.div(local_ex_df.abs().sum(axis=1), axis=0) * 100,
                        label=s_id,
                        index=True,
                    )
                    for s_id, local_ex_df in self.local_explanation.items()
                ]
                local_explanation_section = rc.Block(
                    rc.Heading("Local Explanation of Models", level=2),
                    rc.Select(blocks=blocks),
                )

                # Append the global explanation text and section to the "all_sections" list
                all_sections = all_sections + [
                    global_explanation_section,
                    local_explanation_section,
                ]
            except Exception as e:
                logger.warn(f"Failed to generate Explanations with error: {e}.")
                logger.debug(f"Full Traceback: {traceback.format_exc()}")

        model_description = rc.Text(
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

    def get_explain_predict_fn(self, series_id):
        def _custom_predict(
            data,
            model=self.models[series_id],
            dt_column_name=self.datasets._datetime_column_name,
            target_col=self.original_target_column,
        ):
            """
            data: ForecastDatasets.get_data_at_series(s_id)
            """
            if series_id in self.constant_cols:
                data = data.drop(columns=self.constant_cols[series_id])

            data = data.drop([target_col], axis=1)
            data[dt_column_name] = seconds_to_datetime(
                data[dt_column_name], dt_format=self.spec.datetime_column.format
            )
            data = self.preprocess(data, series_id)
            return model.predict(X=data, n_periods=len(data))

        return _custom_predict
