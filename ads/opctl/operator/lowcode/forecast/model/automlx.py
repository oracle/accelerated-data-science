#!/usr/bin/env python
# Copyright (c) 2023, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import logging
import os
import traceback

import numpy as np
import pandas as pd
import report_creator as rc

from ads.common.decorator.runtime_dependency import runtime_dependency
from ads.opctl import logger
from ads.opctl.operator.lowcode.common.utils import (
    seconds_to_datetime,
)
from ads.opctl.operator.lowcode.forecast.const import (
    AUTOMLX_METRIC_MAP,
    ForecastOutputColumns,
    SpeedAccuracyMode,
    SupportedModels,
)
from ads.opctl.operator.lowcode.forecast.utils import _label_encode_dataframe

from ..operator_config import ForecastOperatorConfig
from .base_model import ForecastOperatorBaseModel
from .forecast_datasets import ForecastDatasets, ForecastOutput

logging.getLogger("report_creator").setLevel(logging.WARNING)
AUTOMLX_N_ALGOS_TUNED = 4
AUTOMLX_DEFAULT_SCORE_METRIC = ['neg_sym_mean_abs_percent_error',
                                'neg_mean_abs_percent_error',
                                'neg_root_mean_squared_error']


class AutoMLXOperatorModel(ForecastOperatorBaseModel):
    """Class representing AutoMLX operator model."""

    def __init__(self, config: ForecastOperatorConfig, datasets: ForecastDatasets):
        super().__init__(config, datasets)
        self.global_explanation = {}
        self.local_explanation = {}
        self.explainability_kwargs = {}

    def set_kwargs(self):
        model_kwargs_cleaned = self.spec.model_kwargs
        model_kwargs_cleaned["n_algos_tuned"] = model_kwargs_cleaned.get(
            "n_algos_tuned", AUTOMLX_N_ALGOS_TUNED
        )
        metric_to_optimize = AUTOMLX_METRIC_MAP.get(self.spec.metric)
        model_kwargs_cleaned["score_metric"] = AUTOMLX_DEFAULT_SCORE_METRIC
        # The first score metric in the list will be the one for which the pipeline optimizes
        if metric_to_optimize is not None:
            model_kwargs_cleaned["score_metric"].remove(metric_to_optimize)
            model_kwargs_cleaned["score_metric"].insert(0, metric_to_optimize)

        model_kwargs_cleaned.pop("task", None)
        time_budget = model_kwargs_cleaned.pop("time_budget", -1)
        model_kwargs_cleaned["preprocessing"] = (
            self.spec.preprocessing.enabled
            or model_kwargs_cleaned.get("preprocessing", True)
        )
        sample_ratio = model_kwargs_cleaned.pop("sample_to_feature_ratio", None)
        if sample_ratio is not None:
            self.explainability_kwargs = {"sample_to_feature_ratio": sample_ratio}
        return model_kwargs_cleaned, time_budget

    def preprocess(self, data, series_id):  # TODO: re-use self.le for explanations
        self.le[series_id], df_encoded = _label_encode_dataframe(
            data,
            no_encode={self.spec.datetime_column.name, self.original_target_column},
        )
        return df_encoded.set_index(self.spec.datetime_column.name)

    @runtime_dependency(
        module="automlx",
        err_msg=(
            "Please run `pip3 install oracle-automlx[forecasting]>=25.3.0` "
            "to install the required dependencies for automlx."
        ),
    )
    @runtime_dependency(
        module="sktime",
        err_msg=(
            "Please run `pip3 install sktime` to install the required dependencies for automlx."
        ),
    )
    def _build_model(self) -> pd.DataFrame:
        import logging

        from automlx import Pipeline, init

        full_data_dict = self.datasets.get_data_by_series()

        self.models = {}
        horizon = self.spec.horizon
        self.spec.confidence_interval_width = self.spec.confidence_interval_width or 0.8
        self.forecast_output = ForecastOutput(
            confidence_interval_width=self.spec.confidence_interval_width,
            horizon=self.spec.horizon,
            target_column=self.original_target_column,
            dt_column=self.spec.datetime_column.name,
        )

        # Clean up kwargs for pass through
        model_kwargs_cleaned, time_budget = self.set_kwargs()

        cpu_count = os.cpu_count()
        try:
            engine_type = model_kwargs_cleaned.pop(
                "engine", "local" if cpu_count <= 4 else "ray"
            )
            engine_opts = (
                None
                if engine_type == "local"
                else {"ray_setup": {"_temp_dir": "/tmp/ray-temp"}}
            )
            init(
                engine=engine_type,
                engine_opts=engine_opts,
                loglevel=logging.CRITICAL,
            )
        except Exception as e:
            logger.info(
                f"Error initializing automlx. Has Ray already been initialized? Skipping. {e}"
            )

        for s_id, df in full_data_dict.items():
            try:
                logger.debug(f"Running automlx on series {s_id}")
                model_kwargs = model_kwargs_cleaned.copy()
                target = self.original_target_column
                self.forecast_output.init_series_output(
                    series_id=s_id, data_at_series=df
                )
                data = self.preprocess(df, s_id)
                data_i = self.drop_horizon(data)
                X_pred = self.get_horizon(data).drop(target, axis=1)

                logger.debug(
                    f"Time Index Monotonic: {data_i.index.is_monotonic_increasing}"
                )

                if self.loaded_models is not None and s_id in self.loaded_models:
                    model = self.loaded_models[s_id]["model"]
                    model_kwargs["model_list"] = [model.selected_model_]
                    model_kwargs["search_space"]={}
                    model_kwargs["search_space"][model.selected_model_] = model.selected_model_params_

                model = Pipeline(
                    task="forecasting",
                    **model_kwargs,
                )
                model.fit(
                    X=data_i.drop(target, axis=1),
                    y=data_i[[target]],
                    time_budget=time_budget,
                )
                logger.debug(f"Selected model: {model.selected_model_}")
                logger.debug(f"Selected model params: {model.selected_model_params_}")
                summary_frame = model.forecast(
                    X=X_pred,
                    periods=horizon,
                    alpha=1 - self.spec.confidence_interval_width,
                )

                fitted_values = model.predict(data_i.drop(target, axis=1))[
                    target
                ].values

                self.models[s_id] = {}
                self.models[s_id]["model"] = model
                self.models[s_id]["le"] = self.le[s_id]
                self.models[s_id]["score"] = self.get_all_metrics(model)

                # In case of Naive model, model.forecast function call does not return confidence intervals.
                if f"{target}_ci_upper" not in summary_frame:
                    summary_frame[f"{target}_ci_upper"] = np.NAN
                if f"{target}_ci_lower" not in summary_frame:
                    summary_frame[f"{target}_ci_lower"] = np.NAN

                if summary_frame[target].isna().all():
                    raise ValueError("The forecasts are completely NaN")

                self.forecast_output.populate_series_output(
                    series_id=s_id,
                    fit_val=fitted_values,
                    forecast_val=summary_frame[target],
                    upper_bound=summary_frame[f"{target}_ci_upper"],
                    lower_bound=summary_frame[f"{target}_ci_lower"],
                )

                self.model_parameters[s_id] = {
                    "framework": SupportedModels.AutoMLX,
                    "time_series_period": model.time_series_period,
                    "selected_model": model.selected_model_,
                    "selected_model_params": model.selected_model_params_,
                }
            except Exception as e:
                new_error = {
                    "model_name": self.spec.model,
                    "error": str(e),
                    "error_trace": traceback.format_exc(),
                }
                if s_id in self.errors_dict:
                    self.errors_dict[s_id]["model_fitting"] = new_error
                else:
                    self.errors_dict[s_id] = {"model_fitting": new_error}
                logger.warning(f"Encountered Error: {e}. Skipping.")
                logger.warning(f"self.errors_dict[s_id]: {self.errors_dict[s_id]}")
                logger.warning(traceback.format_exc())

        logger.debug("===========Forecast Generated===========")

        return self.forecast_output.get_forecast_long()

    def _generate_report(self):
        """
        Generate the report for the automlx model.

        Parameters
        ----------
        None

        Returns
        -------
            - model_description (report-creator.Text): A Text component containing the description of the automlx model.
            - other_sections (List[Union[report-creator.Text, report-creator.Block]]): A list of Text and Blocks components representing various sections of the report.
            - forecast_col_name (str): The name of the forecasted column.
            - train_metrics (bool): A boolean value indicating whether to include train metrics in the report.
            - ds_column_series (pd.Series): The pd.Series object representing the datetime column of the dataset.
            - ds_forecast_col (pd.Series): The pd.Series object representing the forecasted column.
            - ci_col_names (List[str]): A list of column names for the confidence interval in the report.
        """
        selected_models = {}
        models = self.models
        other_sections = []

        if len(self.models) > 0:
            for s_id, artifacts in models.items():
                m = artifacts["model"]
                selected_models[s_id] = {
                    "series_id": s_id,
                    "selected_model": m.selected_model_,
                    "model_params": m.selected_model_params_,
                }
            selected_models_df = pd.DataFrame(
                selected_models.items(), columns=["series_id", "best_selected_model"]
            )
            selected_df = selected_models_df["best_selected_model"].apply(pd.Series)
            if not self.target_cat_col:
                selected_df = selected_df.drop("series_id", axis=1)
            selected_models_section = rc.Block(
                rc.Heading("Selected Models Overview", level=2),
                rc.Text(
                    "The following tables provide information regarding the "
                    "chosen model for each series and the corresponding parameters of the models."
                ),
                rc.DataTable(selected_df, index=True),
            )

            other_sections = [selected_models_section]

        if self.spec.generate_explanations:
            try:
                # If the key is present, call the "explain_model" method
                self.explain_model()

                global_explanation_section = None
                # Convert the global explanation data to a DataFrame
                global_explanation_df = pd.DataFrame(self.global_explanation)

                self.formatted_global_explanation = (
                    global_explanation_df / global_explanation_df.sum(axis=0) * 100
                )

                self.formatted_global_explanation.rename(
                    columns={
                        self.spec.datetime_column.name: ForecastOutputColumns.DATE
                    },
                    inplace=True,
                )

                aggregate_local_explanations = pd.DataFrame()
                for s_id, local_ex_df in self.local_explanation.items():
                    local_ex_df_copy = local_ex_df.copy()
                    local_ex_df_copy["Series"] = s_id
                    aggregate_local_explanations = pd.concat(
                        [aggregate_local_explanations, local_ex_df_copy], axis=0
                    )
                self.formatted_local_explanation = aggregate_local_explanations

                if not self.target_cat_col:
                    self.formatted_global_explanation = (
                        self.formatted_global_explanation.rename(
                            {"Series 1": self.original_target_column},
                            axis=1,
                        )
                    )
                    self.formatted_local_explanation.drop(
                        "Series", axis=1, inplace=True
                    )

                # Create a markdown section for the global explainability
                global_explanation_section = rc.Block(
                    rc.Heading("Global Explanation of Models", level=2),
                    rc.Text(
                        "The following tables provide the feature attribution for the global explainability."
                    ),
                    rc.DataTable(self.formatted_global_explanation, index=True),
                )

                blocks = [
                    rc.DataTable(
                        local_ex_df.div(local_ex_df.abs().sum(axis=1), axis=0) * 100,
                        label=s_id if self.target_cat_col else None,
                        index=True,
                    )
                    for s_id, local_ex_df in self.local_explanation.items()
                ]
                local_explanation_section = rc.Block(
                    rc.Heading("Local Explanation of Models", level=2),
                    rc.Select(blocks=blocks) if len(blocks) > 1 else blocks[0],
                )

                # Append the global explanation text and section to the "other_sections" list
                if global_explanation_section:
                    other_sections.append(global_explanation_section)

                # Append the local explanation text and section to the "other_sections" list
                other_sections = other_sections + [
                    local_explanation_section,
                ]
            except Exception as e:
                logger.warning(f"Failed to generate Explanations with error: {e}.")
                logger.debug(f"Full Traceback: {traceback.format_exc()}")

        model_description = rc.Text(
            "The AutoMLx model automatically preprocesses, selects and engineers "
            "high-quality features in your dataset, which are then provided for further processing."
        )

        return (
            model_description,
            other_sections,
        )

    def get_explain_predict_fn(self, series_id):
        selected_model = self.models[series_id]["model"]

        # If training date, use method below. If future date, use forecast!
        def _custom_predict_fn(
            data,
            model=selected_model,
            dt_column_name=self.datasets._datetime_column_name,
            target_col=self.original_target_column,
            last_train_date=self.datasets.historical_data.get_max_time(),
            horizon_data=self.datasets.get_horizon_at_series(series_id),
        ):
            """
            data: ForecastDatasets.get_data_at_series(s_id)
            """
            data = data.drop(target_col, axis=1)
            data[dt_column_name] = seconds_to_datetime(
                data[dt_column_name], dt_format=self.spec.datetime_column.format
            )
            data = self.preprocess(data, series_id)
            horizon_data = horizon_data.drop(target_col, axis=1)
            horizon_data[dt_column_name] = seconds_to_datetime(
                horizon_data[dt_column_name], dt_format=self.spec.datetime_column.format
            )
            horizon_data = self.preprocess(horizon_data, series_id)

            rows = []
            for i in range(data.shape[0]):
                row = data.iloc[i : i + 1]
                if row.index[0] > last_train_date:
                    X_new = horizon_data.copy()
                    X_new.loc[row.index[0]] = row.iloc[0]
                    row_i = (
                        model.forecast(X=X_new, periods=self.spec.horizon)[[target_col]]
                        .loc[row.index[0]]
                        .values[0]
                    )
                else:
                    row_i = model.predict(X=row).values[0][0]
                rows.append(row_i)
            ret = np.asarray(rows).flatten()
            return ret

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
            columns=list(self.dataset_cols),
        )

        return self.models.get(self.series_id).forecast(
            X=data_temp, periods=data_temp.shape[0]
        )[self.series_id]

    @runtime_dependency(
        module="automlx",
        err_msg=(
            "Please run `python3 -m pip install automlx` to install the required dependencies for model explanation."
        ),
    )
    def explain_model(self):
        """
        Generates explanations for the model using the AutoMLx library.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This function works by generating local explanations for each series in the dataset.
        It uses the ``MLExplainer`` class from the AutoMLx library to generate feature attributions
        for each series. The feature attributions are then stored in the ``self.local_explanation`` dictionary.

        If the accuracy mode is set to AutoMLX, it uses the AutoMLx library to generate explanations.
        Otherwise, it falls back to the default explanation generation method.
        """
        import automlx

        # Loop through each series in the dataset
        for s_id, data_i in self.datasets.get_data_by_series(
            include_horizon=False
        ).items():
            try:
                if self.spec.explanations_accuracy_mode == SpeedAccuracyMode.AUTOMLX:
                    # Use the MLExplainer class from AutoMLx to generate explanations
                    explainer = automlx.MLExplainer(
                        self.models[s_id]["model"],
                        self.datasets.additional_data.get_data_for_series(
                            series_id=s_id
                        )
                        .drop(self.spec.datetime_column.name, axis=1)
                        .head(-self.spec.horizon)
                        if self.spec.additional_data
                        else None,
                        pd.DataFrame(data_i[self.spec.target_column]),
                        task="forecasting",
                        **self.explainability_kwargs,
                    )

                    # Generate explanations for the forecast
                    explanations = explainer.explain_prediction(
                        X=self.datasets.additional_data.get_data_for_series(
                            series_id=s_id
                        )
                        .drop(self.spec.datetime_column.name, axis=1)
                        .tail(self.spec.horizon)
                        if self.spec.additional_data
                        else None,
                        forecast_timepoints=list(range(self.spec.horizon + 1)),
                    )

                    # Convert the explanations to a DataFrame
                    explanations_df = pd.concat(
                        [exp.to_dataframe() for exp in explanations]
                    )
                    explanations_df["row"] = explanations_df.groupby(
                        "Feature"
                    ).cumcount()
                    explanations_df = explanations_df.pivot(
                        index="row", columns="Feature", values="Attribution"
                    )
                    explanations_df = explanations_df.reset_index(drop=True)
                    explanations_df[ForecastOutputColumns.DATE] = (
                        self.datasets.get_horizon_at_series(
                            s_id=s_id
                        )[self.spec.datetime_column.name].reset_index(drop=True)
                    )
                    # Store the explanations in the local_explanation dictionary
                    self.local_explanation[s_id] = explanations_df

                    self.global_explanation[s_id] = dict(
                        zip(
                            self.local_explanation[s_id]
                            .drop(ForecastOutputColumns.DATE, axis=1)
                            .columns,
                            np.nanmean(
                                np.abs(
                                    self.local_explanation[s_id].drop(
                                        ForecastOutputColumns.DATE, axis=1
                                    )
                                ),
                                axis=0,
                            ),
                        )
                    )
                else:
                    # Fall back to the default explanation generation method
                    super().explain_model()
            except Exception as e:
                if s_id in self.errors_dict:
                    self.errors_dict[s_id]["explainer_error"] = str(e)
                    self.errors_dict[s_id]["explainer_error_trace"] = (
                        traceback.format_exc()
                    )
                else:
                    self.errors_dict[s_id] = {
                        "model_name": self.spec.model,
                        "explainer_error": str(e),
                        "explainer_error_trace": traceback.format_exc(),
                    }
                logger.warning(
                    f"Failed to generate explanations for series {s_id} with error: {e}."
                )
                logger.debug(f"Full Traceback: {traceback.format_exc()}")

    def get_all_metrics(self, model):
        trials = model.completed_trials_summary_
        model_params = model.selected_model_params_
        if len(trials) > 0:
            all_metrics = trials[trials.Hyperparameters == model_params][
                "All Metrics"
            ].iloc[0]
        else:
            all_metrics = {}
        reverse_map = {v: k for k, v in AUTOMLX_METRIC_MAP.items()}
        all_metrics = {reverse_map[key]: -1 * value for key, value in all_metrics.items() if key in reverse_map}
        return all_metrics

    def generate_train_metrics(self) -> pd.DataFrame:
        """
        Generate Training Metrics for Automlx
        """
        total_metrics = pd.DataFrame()
        for s_id in self.forecast_output.list_series_ids():
            try:
                metrics = self.models[s_id]["score"]
                metrics_df = pd.DataFrame.from_dict(
                    metrics, orient="index", columns=[s_id]
                )
                logger.warning(
                    "AutoMLX failed to generate training metrics. Recovering validation loss instead"
                )
                total_metrics = pd.concat([total_metrics, metrics_df], axis=1)
            except Exception as e:
                logger.debug(
                    f"Failed to generate training metrics for target_series: {s_id}"
                )
                logger.debug(f"Error: {e}")
        return total_metrics
