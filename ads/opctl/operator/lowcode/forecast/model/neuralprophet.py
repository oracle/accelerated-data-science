#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from torch import Tensor
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    R2Score,
    SymmetricMeanAbsolutePercentageError,
)

from ads.common.decorator.runtime_dependency import (
    OptionalDependency,
    runtime_dependency,
)
from ads.opctl import logger

from ..const import DEFAULT_TRIALS, ForecastOutputColumns, SupportedModels
from ads.opctl.operator.lowcode.forecast.utils import (
    load_pkl,
    write_pkl,
    _select_plot_list,
    _label_encode_dataframe,
)
from ads.opctl.operator.lowcode.common.utils import (
    disable_print,
    enable_print,
    seconds_to_datetime,
)
from .base_model import ForecastOperatorBaseModel
from ..operator_config import ForecastOperatorConfig
from .forecast_datasets import ForecastDatasets, ForecastOutput
import traceback


# def _get_np_metrics_dict(selected_metric):
#     metric_translation = {
#         "mape": MeanAbsolutePercentageError,
#         "smape": SymmetricMeanAbsolutePercentageError,
#         "mae": MeanAbsoluteError,
#         "r2": R2Score,
#         "rmse": MeanSquaredError,
#     }
#     if selected_metric not in metric_translation.keys():
#         logger.warn(
#             f"Could not find the metric: {selected_metric} in torchmetrics. Defaulting to MAE and RMSE"
#         )
#         return {"MAE": MeanAbsoluteError(), "RMSE": MeanSquaredError()}
#     return {selected_metric: metric_translation[selected_metric]()}


@runtime_dependency(
    module="neuralprophet",
    object="NeuralProphet",
    install_from=OptionalDependency.FORECAST,
)
def _fit_model(data, params, additional_regressors, select_metric):
    from neuralprophet import NeuralProphet, set_log_level

    if logger.level > 10:
        set_log_level(logger.level)
        disable_print()

    m = NeuralProphet(**params)
    # m.metrics = _get_np_metrics_dict(select_metric)
    for add_reg in additional_regressors:
        m = m.add_future_regressor(name=add_reg)
    m.fit(df=data)
    accepted_regressors_config = m.config_regressors or dict()
    if hasattr(accepted_regressors_config, "regressors"):
        accepted_regressors_config = accepted_regressors_config.regressors or dict()

    enable_print()
    return m, list(accepted_regressors_config.keys())


class NeuralProphetOperatorModel(ForecastOperatorBaseModel):
    """Class representing NeuralProphet operator model."""

    def __init__(self, config: ForecastOperatorConfig, datasets: ForecastDatasets):
        super().__init__(config=config, datasets=datasets)
        self.forecast_col_name = "yhat1"
        self.loaded_trainers = None
        self.trainers = None

    def _load_model(self):
        try:
            self.loaded_models = load_pkl(self.spec.previous_output_dir + "/model.pkl")
            self.loaded_trainers = load_pkl(
                self.spec.previous_output_dir + "/trainer.pkl"
            )
        except:
            logger.debug("model.pkl/trainer.pkl is not present")

    def set_kwargs(self):
        # Extract the Confidence Interval Width and convert to prophet's equivalent - interval_width
        if self.spec.confidence_interval_width is None:
            quantiles = model_kwargs.get("quantiles", [0.05, 0.95])
            self.spec.confidence_interval_width = float(quantiles[1]) - float(
                quantiles[0]
            )
        else:
            boundaries = round((1 - self.spec.confidence_interval_width) / 2, 2)
            quantiles = [boundaries, self.spec.confidence_interval_width + boundaries]

        model_kwargs = self.spec.model_kwargs
        model_kwargs["quantiles"] = quantiles
        return model_kwargs

    def _train_model(self, i, s_id, df, model_kwargs):
        try:
            self.forecast_output.init_series_output(series_id=s_id, data_at_series=df)

            data = self.preprocess(df, s_id)
            data_i = self.drop_horizon(data)

            if self.loaded_models is not None and s_id in self.loaded_models:
                model = self.loaded_models[s_id]
                accepted_regressors_config = (
                    model.config_regressors.regressors or dict()
                )
                if hasattr(accepted_regressors_config, "regressors"):
                    accepted_regressors_config = (
                        accepted_regressors_config.regressors or dict()
                    )
                self.accepted_regressors[s_id] = list(accepted_regressors_config.keys())
                if self.loaded_trainers is not None and s_id in self.loaded_trainers:
                    model.trainer = self.loaded_trainers[s_id]
            else:
                if self.perform_tuning:
                    model_kwargs = self.run_tuning(data_i, model_kwargs)

                # Build and fit model
                model, self.accepted_regressors[s_id] = _fit_model(
                    data=data_i,
                    params=model_kwargs,
                    additional_regressors=self.additional_regressors,
                    select_metric=None,
                    # select_metric=self.spec.metric,
                )

            logger.debug(
                f"Found the following additional data columns: {self.additional_regressors}"
            )
            if set(self.additional_regressors) - set(self.accepted_regressors[s_id]):
                logger.debug(
                    f"While fitting the model, some additional data may have been "
                    f"discarded. Only using the columns: {self.accepted_regressors[s_id]}"
                )
            # Build future dataframe
            future = data[self.accepted_regressors[s_id] + ["ds"]].reset_index(
                drop=True
            )
            future["y"] = None

            forecast = model.predict(future)
            logger.debug(f"-----------------Model {i}----------------------")
            logger.debug(forecast.tail())

            # TODO; could also extract trend and seasonality?
            cols_to_read = filter(
                lambda x: x.startswith("future_regressor"), forecast.columns
            )
            self.explanations_info[s_id] = forecast[cols_to_read]
            self.explanations_info[s_id]["Date"] = forecast["ds"]
            self.explanations_info[s_id] = self.explanations_info[s_id].set_index(
                "Date"
            )

            self.outputs[s_id] = forecast
            self.forecast_output.populate_series_output(
                series_id=s_id,
                fit_val=self.drop_horizon(forecast["yhat1"]).values,
                forecast_val=self.get_horizon(forecast["yhat1"]).values,
                upper_bound=self.get_horizon(
                    forecast[f"yhat1 {model_kwargs['quantiles'][1]*100}%"]
                ).values,
                lower_bound=self.get_horizon(
                    forecast[f"yhat1 {model_kwargs['quantiles'][0]*100}%"]
                ).values,
            )

            self.models[s_id] = model
            self.trainers[s_id] = model.trainer

            self.model_parameters[s_id] = {
                "framework": SupportedModels.NeuralProphet,
                "config": model.config,
                "config_trend": model.config_trend,
                "config_train": model.config_train,
                "config_seasonality": model.config_seasonality,
                "config_regressors": model.config_regressors,
                "config_ar": model.config_ar,
                "config_events": model.config_events,
                "config_country_holidays": model.config_country_holidays,
                "config_lagged_regressors": model.config_lagged_regressors,
                "config_normalization": model.config_normalization,
                "config_missing": model.config_missing,
                "config_model": model.config_model,
                "config_normalization": model.config_normalization,
                "data_freq": model.data_freq,
                "fitted": model.fitted,
                "data_params": model.data_params,
                "future_periods": model.future_periods,
                "predict_steps": model.predict_steps,
                "highlight_forecast_step_n": model.highlight_forecast_step_n,
                "true_ar_weights": model.true_ar_weights,
            }

            logger.debug("===========Done===========")
        except Exception as e:
            self.errors_dict[s_id] = {"model_name": self.spec.model, "error": str(e)}
            raise e

    def _build_model(self) -> pd.DataFrame:
        full_data_dict = self.datasets.get_data_by_series()
        self.models = dict()
        self.trainers = dict()
        self.outputs = dict()
        self.errors_dict = dict()
        self.explanations_info = dict()
        self.accepted_regressors = dict()
        self.additional_regressors = self.datasets.get_additional_data_column_names()
        model_kwargs = self.set_kwargs()
        self.forecast_output = ForecastOutput(
            confidence_interval_width=self.spec.confidence_interval_width,
            horizon=self.spec.horizon,
            target_column=self.original_target_column,
            dt_column=self.spec.datetime_column.name,
        )

        for i, (s_id, df) in enumerate(full_data_dict.items()):
            self._train_model(i, s_id, df, model_kwargs=model_kwargs.copy())

        # Parallel(n_jobs=-1, require="sharedmem")(
        #     delayed(NeuralProphetOperatorModel._train_model)(self, i, s_id, df, model_kwargs=model_kwargs.copy())
        #     for self, (i, (s_id, df)) in zip(
        #         [self] * len(full_data_dict), enumerate(full_data_dict.items())
        #     )
        # )

        return self.forecast_output.get_forecast_long()

    def run_tuning(self, data, model_kwargs):
        from neuralprophet import NeuralProphet

        def objective(trial):
            params = {
                # 'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
                # 'seasonality_reg': trial.suggest_float('seasonality_reg', 0.1, 500, log=True),
                # 'learning_rate': trial.suggest_float('learning_rate',  0.0001, 0.1, log=True),
                "newer_samples_start": trial.suggest_float(
                    "newer_samples_start", 0.001, 0.999
                ),
                "newer_samples_weight": trial.suggest_float(
                    "newer_samples_weight", 0, 100
                ),
                "changepoints_range": trial.suggest_float(
                    "changepoints_range", 0.8, 0.95
                ),
            }
            # trend_reg, trend_reg_threshold, ar_reg, impute_rolling/impute_linear,
            params.update(model_kwargs)

            folds = NeuralProphet(**params).crossvalidation_split_df(data, k=3)
            test_metrics_total_i = []
            for df_train, df_test in folds:
                m, accepted_regressors = _fit_model(
                    data=df_train,
                    params=params,
                    additional_regressors=self.additional_regressors,
                    select_metric=self.spec.metric,
                )
                df_test = df_test[["y", "ds"] + accepted_regressors]

                test_forecast_i = m.predict(df=df_test)
                fold_metric_i = (
                    m.metrics[self.spec.metric]
                    .forward(
                        Tensor(test_forecast_i["yhat1"]),
                        Tensor(test_forecast_i["y"]),
                    )
                    .item()
                )
                test_metrics_total_i.append(fold_metric_i)
            logger.debug(
                f"----------------------{np.asarray(test_metrics_total_i).mean()}----------------------"
            )
            return np.asarray(test_metrics_total_i).mean()

        study = optuna.create_study(direction="minimize")
        m_params = NeuralProphet().parameters()
        study.enqueue_trial(
            {
                # 'seasonality_mode': m_params['seasonality_mode'],
                # 'seasonality_reg': m_params['seasonality_reg'],
                # 'learning_rate': m_params['learning_rate'],
                "newer_samples_start": m_params["newer_samples_start"],
                "newer_samples_weight": m_params["newer_samples_weight"],
                "changepoints_range": m_params["changepoints_range"],
            }
        )
        study.optimize(
            objective,
            n_trials=self.spec.tuning.n_trials if self.spec.tuning else DEFAULT_TRIALS,
            n_jobs=-1,
        )

        selected_params = study.best_params
        selected_params.update(model_kwargs)
        return selected_params

    def _generate_report(self):
        import report_creator as rc

        series_ids = self.models.keys()
        all_sections = []
        if len(series_ids) > 0:
            try:
                sec1 = _select_plot_list(
                    lambda s_id: self.models[s_id].plot(self.outputs[s_id]),
                    series_ids=series_ids,
                )
                section_1 = rc.Block(
                    rc.Heading("Forecast Overview", level=2),
                    rc.Text(
                        "These plots show your forecast in the context of historical data."
                    ),
                    sec1,
                )
                all_sections = all_sections + [section_1]
            except Exception as e:
                logger.debug(f"Failed to plot with exception: {e.args}")

            try:
                sec2 = _select_plot_list(
                    lambda s_id: self.models[s_id].plot_components(self.outputs[s_id]),
                    series_ids=series_ids,
                )
                section_2 = rc.Block(
                    rc.Heading("Forecast Broken Down by Trend Component", level=2), sec2
                )
                all_sections = all_sections + [section_2]
            except Exception as e:
                logger.debug(f"Failed to plot with exception: {e.args}")

            try:
                sec3 = _select_plot_list(
                    lambda s_id: self.models[s_id].plot_parameters(),
                    series_ids=series_ids,
                )
                section_3 = rc.Block(
                    rc.Heading("Forecast Parameter Plots", level=2), sec3
                )
                all_sections = all_sections + [section_3]
            except Exception as e:
                logger.debug(f"Failed to plot with exception: {e.args}")

            sec5_text = rc.Heading("Neural Prophet Model Parameters", level=2)
            model_states = []
            for i, (s_id, m) in enumerate(self.models.items()):
                model_states.append(
                    pd.Series(
                        m.state_dict(),
                        index=m.state_dict().keys(),
                        name=s_id,
                    )
                )
            all_model_states = pd.concat(model_states, axis=1)
            sec5 = rc.DataTable(all_model_states, index=True)

            all_sections = all_sections + [sec5_text, sec5]

        if self.spec.generate_explanations:
            try:
                # If the key is present, call the "explain_model" method
                self.explain_model()

                # Create a markdown section for the global explainability
                global_explanation_section = rc.Block(
                    rc.Heading("Global Explainability", level=2),
                    rc.Text(
                        "The following tables provide the feature attribution for the global explainability."
                    ),
                    rc.DataTable(self.formatted_global_explanation, index=True),
                )

                blocks = [
                    rc.DataTable(
                        local_ex_df.drop("Series", axis=1),
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
                # Do not fail the whole run due to explanations failure
                logger.warn(f"Failed to generate Explanations with error: {e}.")
                logger.debug(f"Full Traceback: {traceback.format_exc()}")

        model_description = rc.Text(
            "NeuralProphet is an easy to learn framework for interpretable time "
            "series forecasting. NeuralProphet is built on PyTorch and combines "
            "Neural Network and traditional time-series algorithms, inspired by "
            "Facebook Prophet and AR-Net."
        )
        other_sections = all_sections

        return (
            model_description,
            other_sections,
        )

    def _save_model(self, output_dir, storage_options):
        write_pkl(
            obj=self.models,
            filename="model.pkl",
            output_dir=output_dir,
            storage_options=storage_options,
        )
        write_pkl(
            obj=self.trainers,
            filename="trainer.pkl",
            output_dir=output_dir,
            storage_options=storage_options,
        )

    def explain_model(self):
        self.local_explanation = dict()
        global_expl = []
        rename_cols = {
            f"future_regressor_{col}": col
            for col in self.datasets.get_additional_data_column_names()
        }

        for s_id, expl_df in self.explanations_info.items():
            expl_df = expl_df.rename(rename_cols, axis=1)
            # Local Expl
            self.local_explanation[s_id] = self.get_horizon(expl_df).drop(
                ["future_regressors_additive"], axis=1
            )
            self.local_explanation[s_id]["Series"] = s_id
            self.local_explanation[s_id].index.rename(self.dt_column_name, inplace=True)
            # Global Expl
            g_expl = self.drop_horizon(expl_df).mean()
            g_expl.name = s_id
            global_expl.append(g_expl)
        self.global_explanation = pd.concat(global_expl, axis=1)
        self.global_explanation = self.global_explanation.drop(
            index=["future_regressors_additive"], axis=0
        )
        self.formatted_global_explanation = (
            self.global_explanation / self.global_explanation.sum(axis=0) * 100
        )
        self.formatted_local_explanation = pd.concat(self.local_explanation.values())
