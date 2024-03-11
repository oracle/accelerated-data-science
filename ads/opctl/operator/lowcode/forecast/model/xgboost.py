#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from scipy.fft import fft

from ads.opctl import logger

from ads.opctl.operator.lowcode.forecast.utils import _label_encode_dataframe
from ads.opctl.operator.lowcode.common.utils import seconds_to_datetime
from .base_model import ForecastOperatorBaseModel
from ..operator_config import ForecastOperatorConfig
import traceback
from .forecast_datasets import ForecastDatasets, ForecastOutput
from ..const import ForecastOutputColumns, SupportedModels


class XGBoostOperatorModel(ForecastOperatorBaseModel):
    """Class representing XGBoost operator model."""

    def __init__(self, config: ForecastOperatorConfig, datasets: ForecastDatasets):
        super().__init__(config, datasets=datasets)
        self.global_explanation = {}
        self.local_explanation = {}
        self.formatted_global_explanation = None
        self.formatted_local_explanation = None

    def set_kwargs(self):
        # # Extract the Confidence Interval Width and convert to arima's equivalent - alpha
        # if self.spec.confidence_interval_width is None:
        #     self.spec.confidence_interval_width = 1 - self.spec.model_kwargs.get(
        #         "alpha", 0.05
        #     )
        model_kwargs = self.spec.model_kwargs
        # model_kwargs["alpha"] = 1 - self.spec.confidence_interval_width
        # if "error_action" not in model_kwargs.keys():
        #     model_kwargs["error_action"] = "ignore"
        return model_kwargs

    def preprocess(self, data, series_id):  # TODO: re-use self.le for explanations
        self.le[series_id], df_encoded = _label_encode_dataframe(
            data,
        )
        original_y = df_encoded[self.original_target_column].values
        df_encoded["__new_target"] = (
            df_encoded[self.original_target_column].diff().fillna(0)
        )
        # df_encoded['__fft_target'] = np.abs(fft(df_encoded[self.original_target_column].values))
        return df_encoded.drop(self.original_target_column, axis=1), original_y

    def reverse_diff(self, y, diff):
        return diff.cumsum() + y[-1 * (self.spec.horizon + 1) : -self.spec.horizon]

    def deconstruct_timestamp(self, df, dt_col_name):
        dt_col = df[dt_col_name]
        df["__Year"] = dt_col.dt.year
        df["__Month"] = dt_col.dt.month
        df["__Day"] = dt_col.dt.day
        df["__DOW"] = dt_col.dt.dayofweek
        df["__Week"] = dt_col.dt.week
        return df.drop(dt_col_name, axis=1)

    def add_lag(self, df, nlags=5):
        for i in range(nlags):
            df[f"__lag{i}"] = df[self.original_target_column].shift(i)
        return df

    def _train_model(self, data_long, model_kwargs):
        """Trains the ARIMA model for a given series of the dataset.

        Parameters
        ----------
        df: pd.DataFrame
            The dataframe containing the target data
        """

        # data_clean = self.deconstruct_timestamp(data_long.reset_index(), self.spec.datetime_column.name)
        # data_clean = self.preprocess(data_clean, series_id="1")
        # model = XGBRegressor(objective='reg:squarederror', n_estimators=10000)

        # X, y = data_clean.drop(self.original_target_column, axis=1), data_clean[self.original_target_column]
        # model.fit(X, y)
        # data_wide = self.datasets.format_wide()

        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split
        import optuna

        for s_id in self.datasets.list_series_ids():
            data = self.datasets.get_data_at_series(s_id, include_horizon=True)
            self.forecast_output.init_series_output(series_id=s_id, data_at_series=data)
            data_clean = self.deconstruct_timestamp(
                data, self.spec.datetime_column.name
            )
            data_clean, original_y = self.preprocess(data_clean, series_id=s_id)
            # data_clean = self.add_lag(data_clean)

            horizon = self.get_horizon(data_clean)
            historical = self.drop_horizon(data_clean)

            X = historical.drop("__new_target", axis=1)
            y = historical["__new_target"]

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if self.spec.tuning is not None:

                def objective(trial):
                    params = {
                        "objective": "reg:squarederror",
                        "n_estimators": 1000,
                        "verbosity": 0,
                        "learning_rate": trial.suggest_float(
                            "learning_rate", 1e-3, 0.1, log=True
                        ),
                        "max_depth": trial.suggest_int("max_depth", 1, 10),
                        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                        "colsample_bytree": trial.suggest_float(
                            "colsample_bytree", 0.05, 1.0
                        ),
                        "min_child_weight": trial.suggest_int(
                            "min_child_weight", 1, 20
                        ),
                    }

                    model = XGBRegressor(**params)
                    model.fit(X_train, y_train, verbose=False)
                    predictions = model.predict(X_val)
                    rmse = mean_squared_error(y_val, predictions, squared=False)
                    return rmse

                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=self.spec.tuning.n_trials)

                model = XGBRegressor(
                    objective="reg:squarederror", n_estimators=1000, **study.best_params
                )
            else:
                model = XGBRegressor(objective="reg:squarederror", n_estimators=1000)
            model.fit(X, y)

            X_pred = horizon.drop("__new_target", axis=1)
            pred = model.predict(X_pred)
            yhat = self.reverse_diff(original_y, model.predict(X_pred))

            fitted_values = model.predict(X)

            forecast = pd.DataFrame(yhat, columns=["yhat"])
            forecast["yhat_lower"] = None
            forecast["yhat_upper"] = None
            logger.debug(f"-----------------Model {s_id}----------------------")
            logger.debug(forecast[["yhat", "yhat_lower", "yhat_upper"]].tail())

            self.forecast_output.populate_series_output(
                series_id=s_id,
                fit_val=fitted_values,
                forecast_val=self.get_horizon(forecast["yhat"]).values,
                upper_bound=self.get_horizon(forecast["yhat_upper"]).values,
                lower_bound=self.get_horizon(forecast["yhat_lower"]).values,
            )

            self.models[s_id] = model

            params = model.get_params()
            self.model_parameters[s_id] = {
                "framework": SupportedModels.XGBoost,
                **params,
            }

            logger.debug("===========Done===========")
            # except Exception as e:
            #     self.errors_dict[s_id] = {"model_name": self.spec.model, "error": str(e)}

    def _build_model(self) -> pd.DataFrame:
        data_long = self.datasets.get_all_data_long(include_horizon=False)
        self.models = dict()
        model_kwargs = self.set_kwargs()
        self.forecast_output = ForecastOutput(
            confidence_interval_width=self.spec.confidence_interval_width,
            horizon=self.spec.horizon,
            target_column=self.original_target_column,
            dt_column=self.spec.datetime_column.name,
        )
        self._train_model(data_long, model_kwargs)

        return self.forecast_output.get_forecast_long()

    def _generate_report(self):
        """The method that needs to be implemented on the particular model level."""
        import datapane as dp

        sec5_text = dp.Text(f"## XGBoost Model Parameters")
        blocks = [
            dp.DataTable(
                pd.DataFrame(m.get_xgb_params(), index=["values"]).T,
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
                self.formatted_global_explanation = (
                    self.formatted_global_explanation.rename(
                        {self.spec.datetime_column.name: ForecastOutputColumns.DATE},
                        axis=1,
                    )
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
            "XGBoost leverages bagging and boosting to create complex models from simpler parts."
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
            data = data.drop([target_col], axis=1)
            data[dt_column_name] = seconds_to_datetime(
                data[dt_column_name], dt_format=self.spec.datetime_column.format
            )
            data = self.preprocess(data, series_id)
            return model.predict(X=data, n_periods=len(data))

        return _custom_predict
