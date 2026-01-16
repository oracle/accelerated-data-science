#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import traceback
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import shap

from ads.common.decorator import runtime_dependency
from ads.opctl import logger
from .base_model import ForecastOperatorBaseModel
from .forecast_datasets import ForecastDatasets, ForecastOutput
from ..const import ForecastOutputColumns, SpeedAccuracyMode
from ..operator_config import ForecastOperatorConfig


class MLForecastBaseModel(ForecastOperatorBaseModel, ABC):
    def __init__(self, config: ForecastOperatorConfig, datasets: ForecastDatasets):
        super().__init__(config=config, datasets=datasets)
        self.global_explanation = {}
        self.local_explanation = {}
        self.formatted_global_explanation = None
        self.formatted_local_explanation = None
        self.date_col = config.spec.datetime_column.name
        self.data_train = self.datasets.get_all_data_long(include_horizon=False)
        self.data_test = self.datasets.get_all_data_long_forecast_horizon()
        self.full_dataset_with_prediction = None
        self.model_name = ""
        self.model_description = ""

    @runtime_dependency(
        module="mlforecast",
        err_msg="MLForecast is not installed, please install it with 'pip install mlforecast'",
    )
    def set_model_config(self, freq, model_kwargs):
        from mlforecast.lag_transforms import ExpandingMean, RollingMean
        from mlforecast.target_transforms import Differences
        seasonal_map = {
            "H": 24,
            "D": 7,
            "W": 52,
            "M": 12,
            "Q": 4,
        }
        sp = seasonal_map.get(freq.upper(), 7)
        series_lengths = self.data_train.groupby(ForecastOutputColumns.SERIES).size()
        min_len = series_lengths.min()
        logger.info(f"Minimum series has {min_len} datapoints")
        max_allowed = min(min_len - sp, min_len // 2)
        default_lags = []
        for l in [1, sp, 2 * sp]:
            default_lags.append(l)
            if sum(default_lags) > max_allowed:
                default_lags.pop()
                break
        lags = model_kwargs.get("lags", default_lags)

        default_roll = 2 * sp
        roll = model_kwargs.get("RollingMean", default_roll)

        default_diff = sp if sp <= max_allowed else None
        diff = model_kwargs.get("Differences", default_diff)

        return {
            "target_transforms": [Differences([diff])],
            "lags": lags,
            "lag_transforms": {
                1: [ExpandingMean()],
                sp: [RollingMean(window_size=roll, min_samples=1)]
            }
        }

    @abstractmethod
    def _train_model(self, data_train, data_test, model_kwargs) -> pd.DataFrame:
        """
        Build the model.
        The method that needs to be implemented on the particular model level.
        """

    @abstractmethod
    def get_model_kwargs(self) -> pd.DataFrame:
        """
        Build the model.
        The method that needs to be implemented on the particular model level.
        """

    def _build_model(self) -> pd.DataFrame:
        self.models = {}
        self.forecast_output = ForecastOutput(
            confidence_interval_width=self.spec.confidence_interval_width,
            horizon=self.spec.horizon,
            target_column=self.original_target_column,
            dt_column=self.date_col,
        )
        self._train_model(self.data_train, self.data_test, self.get_model_kwargs())
        return self.forecast_output.get_forecast_long()

    def explain_model(self):
        self.local_explanation = {}
        global_expl = []

        import numpy as np
        for shap_vals in self.shap_data:
            s_id = shap_vals["series_id"]
            shap_df = shap_vals["shap_values"]
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

        sec2_text = rc.Block(
            rc.Heading(f"{self.model_name} Model Parameters", level=2),
            rc.Text(f"These are the parameters used for the {self.model_name} model."),
        )

        k, v = next(iter(self.model_parameters.items()))
        sec_2 = rc.Html(
            pd.DataFrame(list(v.items())).to_html(index=False, header=False),
        )

        all_sections = [sec2_text, sec_2]

        if self.spec.generate_explanations:
            try:
                # If the key is present, call the "explain_model" method
                self._generate_shap_tree_explanations()
                self.explain_model()

                global_explanation_section, local_explanation_section = self.generate_explanation_report_from_data()

                # Append the global explanation text and section to the "all_sections" list
                all_sections = all_sections + [
                    global_explanation_section,
                    local_explanation_section,
                ]
            except Exception as e:
                # Do not fail the whole run due to explanations failure
                logger.error(f"Failed to generate Explanations with error: {e}.")
                logger.error(f"Full Traceback: {traceback.format_exc()}")
                self.errors_dict["explainer_error"] = str(e)
                self.errors_dict["explainer_error_error"] = traceback.format_exc()
        model_description = rc.Text(self.model_description)

        return model_description, all_sections

    def _generate_shap_tree_explanations(self):
        """Generate SHAP explanations for the model (handles both single and recursive models)."""
        dataset = self.full_dataset_with_prediction
        model_kwargs = self.spec.model_kwargs
        try:
            # Preprocess data to get features
            model_columns = [
                                ForecastOutputColumns.SERIES
                            ] + dataset.select_dtypes(exclude=["object"]).columns.to_list()

            X, y = self.fcst.preprocess(
                df=dataset[model_columns],
                id_col=ForecastOutputColumns.SERIES,
                time_col=self.date_col,
                target_col=self.original_target_column,
                static_features=model_kwargs.get("static_features", []),
                return_X_y=True,
            )
            X[ForecastOutputColumns.SERIES] = dataset[ForecastOutputColumns.SERIES][len(dataset) - len(X):]
            X[self.dt_column_name] = dataset[self.dt_column_name][len(dataset) - len(X):]

            # Get the forecast models
            forecast_models = self.fcst.models_["forecast"]
            self.shap_data = []

            shap_cols = [col for col in X.columns.tolist() if
                         col not in [ForecastOutputColumns.SERIES, self.dt_column_name]]

            ratio = SpeedAccuracyMode.ratio[self.spec.explanations_accuracy_mode]
            logger.debug(
                "Using multiple model for all horizons"
                if isinstance(forecast_models, list)
                else "Using single model for all horizons"
            )

            for s_id in self.datasets.list_series_ids():
                series_df = X[X[ForecastOutputColumns.SERIES] == s_id]
                series_df = series_df.tail(
                    max(int(len(series_df) * ratio), 5)
                ).reset_index(drop=True)

                shap_values = self._compute_shap_values(
                    series_df=series_df,
                    forecast_models=forecast_models,
                    shap_cols=shap_cols,
                )

                shap_df = pd.DataFrame(shap_values, columns=shap_cols)

                aggregated_shap_df = self._aggregate_shap(
                    shap_df=shap_df,
                    series_df=series_df,
                )

                self.shap_data.append({
                    "series_id": s_id,
                    "shap_values": aggregated_shap_df,
                })

            self.shap_data[0]['shap_values'].to_csv("shap_df", index=False)

        except Exception as e:
            logger.error(f"Failed to generate SHAP explanations: {e}")
            logger.error(traceback.format_exc())
            self.errors_dict["shap_explainer_error"] = str(e)

    def _compute_shap_values(
            self,
            series_df: pd.DataFrame,
            forecast_models,
            shap_cols,
    ):
        shap_chunks = []

        # Multi-model (one per horizon)
        if isinstance(forecast_models, list):
            base_explainer = shap.TreeExplainer(forecast_models[0])

            training_df = series_df[:len(series_df) - len(forecast_models)]
            horizon_df = series_df.tail(len(forecast_models))

            # training SHAP
            if len(training_df) > 0:
                shap_chunks.append(
                    base_explainer.shap_values(training_df[shap_cols])
                )

            # horizon SHAP
            for ind, md in enumerate(forecast_models):
                h_explainer = shap.TreeExplainer(md)
                shap_chunks.append(
                    h_explainer.shap_values(
                        horizon_df[shap_cols].iloc[[ind]]
                    )
                )

        # Single-model
        else:
            explainer = shap.TreeExplainer(forecast_models)
            shap_chunks.append(
                explainer.shap_values(series_df[shap_cols])
            )

        return np.concatenate(shap_chunks, axis=0)

    def _map_feature_to_base(self, feature_name):
        if feature_name.startswith(("lag", "rolling", "expanding")):
            return self.original_target_column
        if feature_name in ["year", "month", "day", "dayofweek", "dayofyear"]:
            return ForecastOutputColumns.DATE_IMPORTANCE
        return feature_name

    def _aggregate_shap(self, shap_df: pd.DataFrame, series_df: pd.DataFrame):
        aggregated_shap = {}

        for col in shap_df.columns:
            base_col = self._map_feature_to_base(col)
            aggregated_shap.setdefault(base_col, 0)
            aggregated_shap[base_col] += shap_df[col]

        aggregated_shap_df = pd.DataFrame(aggregated_shap)

        aggregated_shap_df[ForecastOutputColumns.SERIES] = (
            series_df[ForecastOutputColumns.SERIES].values
        )
        aggregated_shap_df[self.dt_column_name] = (
            series_df[self.dt_column_name].values
        )

        cls = self.full_dataset_with_prediction.columns.tolist() + [ForecastOutputColumns.DATE_IMPORTANCE]
        cls = [c for c in cls if c in aggregated_shap_df.columns]
        aggregated_shap_df = aggregated_shap_df[cls]
        aggregated_shap_df.set_index(self.dt_column_name, inplace=True)

        return aggregated_shap_df
