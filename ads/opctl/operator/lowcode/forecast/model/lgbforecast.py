#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import logging
import traceback

import pandas as pd
import shap
from ads.common.decorator import runtime_dependency
from ads.opctl import logger
from .forecast_datasets import ForecastDatasets, ForecastOutput
from .ml_forecast import MLForecastBaseModel
from ..const import ForecastOutputColumns, SupportedModels, SpeedAccuracyMode
from ..operator_config import ForecastOperatorConfig


class LGBForecastOperatorModel(MLForecastBaseModel):
    """Class representing MLForecast operator model."""

    def __init__(self, config: ForecastOperatorConfig, datasets: ForecastDatasets):
        super().__init__(config=config, datasets=datasets)

    def get_model_kwargs(self):
        """
        Returns the model parameters.
        """
        model_kwargs = self.spec.model_kwargs

        upper_quantile = round(0.5 + self.spec.confidence_interval_width / 2, 2)
        lower_quantile = round(0.5 - self.spec.confidence_interval_width / 2, 2)

        model_kwargs["lower_quantile"] = lower_quantile
        model_kwargs["upper_quantile"] = upper_quantile
        return model_kwargs


    def preprocess(self, df, series_id):
        pass

    @runtime_dependency(
        module="mlforecast",
        err_msg="MLForecast is not installed, please install it with 'pip install mlforecast'",
    )
    @runtime_dependency(
        module="lightgbm",
        err_msg="lightgbm is not installed, please install it with 'pip install lightgbm'",
    )
    def _train_model(self, data_train, data_test, model_kwargs):
        import lightgbm as lgb
        from mlforecast import MLForecast
        try:

            lgb_params = {
                "verbosity": model_kwargs.get("verbosity", -1),
                "num_leaves": model_kwargs.get("num_leaves", 512),
            }

            data_freq = self.datasets.get_datetime_frequency()

            additional_data_params = self.set_model_config(data_freq, model_kwargs)

            fcst = MLForecast(
                models={
                    "forecast": lgb.LGBMRegressor(**lgb_params),
                    "upper": lgb.LGBMRegressor(
                        **lgb_params,
                        objective="quantile",
                        alpha=model_kwargs["upper_quantile"],
                    ),
                    "lower": lgb.LGBMRegressor(
                        **lgb_params,
                        objective="quantile",
                        alpha=model_kwargs["lower_quantile"],
                    ),
                },
                freq=data_freq,
                date_features=['year', 'month', 'day', 'dayofweek', 'dayofyear'],
                **additional_data_params,
            )

            num_models = model_kwargs.get("recursive_models", False)

            self.model_columns = [
                ForecastOutputColumns.SERIES
            ] + data_train.select_dtypes(exclude=["object"]).columns.to_list()
            fcst.fit(
                data_train[self.model_columns],
                static_features=model_kwargs.get("static_features", []),
                id_col=ForecastOutputColumns.SERIES,
                time_col=self.date_col,
                target_col=self.spec.target_column,
                fitted=True,
                max_horizon=None if num_models is False else self.spec.horizon,
            )

            test_df = pd.concat(
                [
                    data_test[self.model_columns],
                    fcst.get_missing_future(
                        h=self.spec.horizon, X_df=data_test[self.model_columns]
                    ),
                ],
                axis=0,
                ignore_index=True,
            )
            test_df.fillna(0)

            self.outputs = fcst.predict(
                h=self.spec.horizon,
                X_df=test_df,
            )
            # print(f'Forecast output test: {test_df.head(10)} \n {self.outputs.head(10)}')
            self.fcst = fcst
            print(
                f"train_df: {data_train.dtypes} {data_train.index} \n test_df:  {test_df.head(2)} \n {test_df.dtypes} {test_df.index}")

            self.fitted_values = fcst.forecast_fitted_values()
            for s_id in self.datasets.list_series_ids():
                self.forecast_output.init_series_output(
                    series_id=s_id,
                    data_at_series=self.datasets.get_data_at_series(s_id),
                )

                self.forecast_output.populate_series_output(
                    series_id=s_id,
                    fit_val=self.fitted_values[
                        self.fitted_values[ForecastOutputColumns.SERIES] == s_id
                    ].forecast.values,
                    forecast_val=self.outputs[
                        self.outputs[ForecastOutputColumns.SERIES] == s_id
                    ].forecast.values,
                    upper_bound=self.outputs[
                        self.outputs[ForecastOutputColumns.SERIES] == s_id
                    ].upper.values,
                    lower_bound=self.outputs[
                        self.outputs[ForecastOutputColumns.SERIES] == s_id
                    ].lower.values,
                )

                one_step_model = fcst.models_['forecast'][0] if isinstance(fcst.models_['forecast'], list) else \
                fcst.models_['forecast']
                self.model_parameters[s_id] = {
                    "framework": SupportedModels.LGBForecast,
                    **lgb_params,
                    **one_step_model.get_params(),
                }

            if self.spec.generate_explanations:
                predictions_df = self.outputs.sort_values(
                    by=[ForecastOutputColumns.SERIES, ForecastOutputColumns.DATE]).reset_index(drop=True)
                test_df = test_df.sort_values(
                    by=[ForecastOutputColumns.SERIES, ForecastOutputColumns.DATE]).reset_index(drop=True)
                test_df[self.spec.target_column] = predictions_df['forecast']
                full_dataset = pd.concat([data_train, test_df], ignore_index=True, axis=0)
                print(f"full_df : {full_dataset.head(10)} :: {full_dataset.dtypes} :: {full_dataset.index}")
                self._generate_shap_explanations(full_dataset, model_kwargs)

            logger.debug("===========Done===========")

        except Exception as e:
            self.errors_dict[self.spec.model] = {
                "model_name": self.spec.model,
                "error": str(e),
                "error_trace": traceback.format_exc(),
            }
            logger.warning(f"Encountered Error: {e}. Skipping.")
            logger.warning(traceback.format_exc())
            raise e

    def explain_model(self):
        self.local_explanation = {}
        global_expl = []

        import numpy as np
        for shap_vals in self.shap_data:
            s_id = shap_vals["series_id"]
            shap_df = shap_vals["shap_values"]
            print(f"shap_df: {shap_df.head(2)}, \n {shap_df.index} \n {shap_df.columns} \n {shap_df.dtypes}")
            # Local Expl
            self.local_explanation[s_id] = self.get_horizon(shap_df)
            self.local_explanation[s_id]["Series"] = s_id
            self.local_explanation[s_id].index.rename(self.dt_column_name, inplace=True)
            # Global expl
            g_expl = shap_df if len(shap_df) <= self.spec.horizon else self.drop_horizon(shap_df)
            g_expl = g_expl.drop(columns=[ForecastOutputColumns.SERIES])
            g_expl = g_expl.mean()
            g_expl.name = s_id
            global_expl.append(np.abs(g_expl))
        self.global_explanation = pd.concat(global_expl, axis=1)
        self.formatted_global_explanation = (
                self.global_explanation / self.global_explanation.sum(axis=0) * 100
        )
        self.formatted_local_explanation = pd.concat(self.local_explanation.values())

    def _generate_report(self):
        """
        Generates the report for the model
        """
        import report_creator as rc

        logging.getLogger("report_creator").setLevel(logging.WARNING)

        # Section 2: LGBForecast Model Parameters
        sec2_text = rc.Block(
            rc.Heading("LGBForecast Model Parameters", level=2),
            rc.Text("These are the parameters used for the LGBForecast model."),
        )

        k, v = next(iter(self.model_parameters.items()))
        sec_2 = rc.Html(
            pd.DataFrame(list(v.items())).to_html(index=False, header=False),
        )

        all_sections = [sec2_text, sec_2]

        if self.spec.generate_explanations:
            try:
                # If the key is present, call the "explain_model" method
                self.explain_model()

                global_explanation_section, local_explanation_section = self.generate_explanation_report_from_data()

                # Append the global explanation text and section to the "all_sections" list
                all_sections = all_sections + [
                    global_explanation_section,
                    local_explanation_section,
                ]
            except Exception as e:
                # Do not fail the whole run due to explanations failure
                print(f"Failed to generate Explanations with error: {e}.")
                print(f"Full Traceback: {traceback.format_exc()}")
                self.errors_dict["explainer_error"] = str(e)
                self.errors_dict["explainer_error_error"] = traceback.format_exc()
        model_description = rc.Text(
            "LGBForecast uses mlforecast framework to perform time series forecasting using machine learning models"
            "with the option to scale to massive amounts of data using remote clusters."
            "Fastest implementations of feature engineering for time series forecasting in Python."
            "Support for exogenous variables and static covariates."
        )

        return model_description, all_sections

    def _generate_shap_explanations(self, data_train, model_kwargs):
        """Generate SHAP explanations for the model (handles both single and recursive models)."""

        try:
            # Preprocess data to get features
            X, y = self.fcst.preprocess(
                df=data_train[self.model_columns],
                id_col=ForecastOutputColumns.SERIES,
                time_col=self.date_col,
                target_col=self.original_target_column,
                static_features=model_kwargs.get("static_features", []),
                return_X_y=True,
            )
            print(f"Feature matrix shape: {X.head(10)}  :: \n {y}")
            X[ForecastOutputColumns.SERIES] = data_train[ForecastOutputColumns.SERIES][len(data_train) - len(X):]
            X[ForecastOutputColumns.SERIES] = data_train[ForecastOutputColumns.DATE][len(data_train) - len(X):]
            # import numpy as np
            # np.savetxt('output.csv', y, delimiter=',')

            # Get the forecast models
            forecast_models = self.fcst.models_["forecast"]
            self.shap_data = []

            def map_feature_to_base(feature_name):
                if feature_name.startswith(("lag", "rolling", "expanding")):
                    return self.original_target_column
                return feature_name

            if isinstance(forecast_models, list):
                logger.debug(
                    f"Using recursive models: {len(forecast_models)} models for horizons 1-{len(forecast_models)}")

                # Store SHAP values for each horizon model
                all_shap_values = []
                self.feature_names = data_train.columns.tolist()

                for horizon_idx, model in enumerate(forecast_models):
                    logger.debug(f"Generating SHAP values for horizon {horizon_idx + 1}")
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(
                        data_train[data_train[ForecastOutputColumns.SERIES] == horizon_idx])
                    all_shap_values.append(shap_vals)

                # Store SHAP data for each horizon
                self.shap_data = {
                    'shap_values_per_horizon': all_shap_values,
                }

            else:
                print("Using single model for all horizons")
                # Single model case
                shap_cols = [col for col in X.columns.tolist() if
                             col not in [ForecastOutputColumns.SERIES, ForecastOutputColumns.DATE]]
                print(f"shap_cols : {shap_cols}")
                print(
                    f"Calculating explanations using {self.spec.explanations_accuracy_mode} mode"
                )
                ratio = SpeedAccuracyMode.ratio[self.spec.explanations_accuracy_mode]
                for s_id in self.datasets.list_series_ids():
                    series_df = X[X[ForecastOutputColumns.SERIES] == s_id]
                    series_df = series_df.tail(
                        max(int(len(series_df) * ratio), 5)
                    ).reset_index(drop=True)
                    explainer = shap.TreeExplainer(forecast_models)
                    shap_values = explainer.shap_values(series_df[shap_cols])
                    shap_df = pd.DataFrame(shap_values, columns=shap_cols)

                    aggregated_shap = {}

                    for col in shap_cols:
                        base_col = map_feature_to_base(col)
                        aggregated_shap.setdefault(base_col, 0)
                        aggregated_shap[base_col] += shap_df[col]

                    aggregated_shap_df = pd.DataFrame(aggregated_shap)
                    aggregated_shap_df[ForecastOutputColumns.SERIES] = series_df[ForecastOutputColumns.SERIES].values
                    aggregated_shap_df[ForecastOutputColumns.DATE] = series_df[ForecastOutputColumns.DATE].values
                    aggregated_shap_df = aggregated_shap_df[data_train.columns.tolist()]
                    aggregated_shap_df.set_index(ForecastOutputColumns.DATE, inplace=True)

                    self.shap_data.append({
                        'series_id': s_id,
                        'shap_values': aggregated_shap_df,
                    })
            self.shap_data[0]['shap_values'].to_csv("shap_df", index=False)
            print(f"SHAP explanations : {self.shap_data}")

        except Exception as e:
            print(f"Failed to generate SHAP explanations: {e}")
            print(traceback.format_exc())
            self.errors_dict["shap_explainer_error"] = str(e)
