#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np
import optuna
import pandas as pd
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

from ..const import DEFAULT_TRIALS, ForecastOutputColumns
from .. import utils
from .base_model import ForecastOperatorBaseModel
from ..operator_config import ForecastOperatorConfig
from .forecast_datasets import ForecastDatasets, ForecastOutput
import traceback


def _get_np_metrics_dict(selected_metric):
    metric_translation = {
        "mape": MeanAbsolutePercentageError,
        "smape": SymmetricMeanAbsolutePercentageError,
        "mae": MeanAbsoluteError,
        "r2": R2Score,
        "rmse": MeanSquaredError,
    }
    if selected_metric not in metric_translation.keys():
        logger.warn(
            f"Could not find the metric: {selected_metric} in torchmetrics. Defaulting to MAE and RMSE"
        )
        return {"MAE": MeanAbsoluteError(), "RMSE": MeanSquaredError()}
    return {selected_metric: metric_translation[selected_metric]()}


@runtime_dependency(
    module="neuralprophet",
    object="NeuralProphet",
    install_from=OptionalDependency.FORECAST,
)
def _fit_model(data, params, additional_regressors, select_metric):
    from neuralprophet import NeuralProphet

    m = NeuralProphet(**params)
    m.metrics = _get_np_metrics_dict(select_metric)
    for add_reg in additional_regressors:
        m = m.add_future_regressor(name=add_reg)
    m.fit(df=data)
    accepted_regressors_config = m.config_regressors or dict()
    return m, list(accepted_regressors_config.keys())


class NeuralProphetOperatorModel(ForecastOperatorBaseModel):
    """Class representing NeuralProphet operator model."""

    def __init__(self, config: ForecastOperatorConfig, datasets: ForecastDatasets):
        super().__init__(config=config, datasets=datasets)
        self.train_metrics = True
        self.forecast_col_name = "yhat1"

    def _build_model(self) -> pd.DataFrame:
        from neuralprophet import NeuralProphet

        full_data_dict = self.datasets.full_data_dict
        models = []
        outputs = dict()
        outputs_legacy = []

        # Extract the Confidence Interval Width and
        # convert to neural prophets equivalent - quantiles
        model_kwargs = self.spec.model_kwargs

        if self.spec.confidence_interval_width is None:
            quantiles = model_kwargs.get("quantiles", [0.05, 0.95])
            self.spec.confidence_interval_width = float(quantiles[1]) - float(
                quantiles[0]
            )
        else:
            boundaries = round((1 - self.spec.confidence_interval_width) / 2, 2)
            quantiles = [boundaries, self.spec.confidence_interval_width + boundaries]

        model_kwargs["quantiles"] = quantiles
        self.forecast_output = ForecastOutput(
            confidence_interval_width=self.spec.confidence_interval_width
        )

        for i, (target, df) in enumerate(full_data_dict.items()):
            le, df_encoded = utils._label_encode_dataframe(
                df, no_encode={self.spec.datetime_column.name, target}
            )
            model_kwargs_i = model_kwargs.copy()

            # format the dataframe for this target. Dropping NA on target[df] will remove all future data
            df_clean = self._preprocess(
                df_encoded,
                self.spec.datetime_column.name,
                self.spec.datetime_column.format,
            )
            data_i = df_clean[df_clean[target].notna()]
            data_i.rename({target: "y"}, axis=1, inplace=True)

            # Assume that all columns passed in should be used as additional data
            additional_regressors = set(data_i.columns) - {"y", "ds"}
            training_data = data_i[["y", "ds"] + list(additional_regressors)]

            if self.perform_tuning:

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
                    params.update(model_kwargs_i)

                    folds = NeuralProphet(**params).crossvalidation_split_df(
                        data_i, k=3
                    )
                    test_metrics_total_i = []
                    for df_train, df_test in folds:
                        m, accepted_regressors = _fit_model(
                            data=df_train,
                            params=params,
                            additional_regressors=additional_regressors,
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
                    n_trials=self.spec.tuning.n_trials
                    if self.spec.tuning
                    else DEFAULT_TRIALS,
                    n_jobs=-1,
                )

                selected_params = study.best_params
                selected_params.update(model_kwargs_i)
                model_kwargs_i = selected_params

            # Build and fit model
            model, accepted_regressors = _fit_model(
                data=training_data,
                params=model_kwargs_i,
                additional_regressors=additional_regressors,
                select_metric=self.spec.metric,
            )
            logger.debug(
                f"Found the following additional data columns: {additional_regressors}"
            )
            logger.debug(
                f"While fitting the model, some additional data may have been "
                f"discarded. Only using the columns: {accepted_regressors}"
            )

            # Build future dataframe
            future = df_clean.reset_index(drop=True)
            future["y"] = None
            future = future[["y", "ds"] + list(accepted_regressors)]

            # Forecast model and collect outputs
            forecast = model.predict(future)
            logger.debug(f"-----------------Model {i}----------------------")
            logger.debug(forecast.tail())
            models.append(model)
            outputs[target] = forecast
            outputs_legacy.append(forecast)

        self.models = models
        self.outputs = outputs_legacy

        logger.debug("===========Done===========")

        # Merge the outputs from each model into 1 df with all outputs by target and category
        col = self.original_target_column
        output_col = pd.DataFrame()
        yhat_upper_name = ForecastOutputColumns.UPPER_BOUND
        yhat_lower_name = ForecastOutputColumns.LOWER_BOUND
        for cat in self.categories:
            output_i = pd.DataFrame()

            output_i["Date"] = outputs[f"{col}_{cat}"]["ds"]
            output_i["Series"] = cat
            output_i[f"input_value"] = full_data_dict[f"{col}_{cat}"][f"{col}_{cat}"]

            output_i[f"fitted_value"] = float("nan")
            output_i[f"forecast_value"] = float("nan")
            output_i[yhat_lower_name] = float("nan")
            output_i[yhat_upper_name] = float("nan")

            output_i.iloc[
                : -self.spec.horizon, output_i.columns.get_loc(f"fitted_value")
            ] = (outputs[f"{col}_{cat}"]["yhat1"].iloc[: -self.spec.horizon].values)
            output_i.iloc[
                -self.spec.horizon :,
                output_i.columns.get_loc(f"forecast_value"),
            ] = (
                outputs[f"{col}_{cat}"]["yhat1"].iloc[-self.spec.horizon :].values
            )
            output_i.iloc[
                -self.spec.horizon :,
                output_i.columns.get_loc(yhat_upper_name),
            ] = (
                outputs[f"{col}_{cat}"][f"yhat1 {quantiles[1]*100}%"]
                .iloc[-self.spec.horizon :]
                .values
            )
            output_i.iloc[
                -self.spec.horizon :,
                output_i.columns.get_loc(yhat_lower_name),
            ] = (
                outputs[f"{col}_{cat}"][f"yhat1 {quantiles[0]*100}%"]
                .iloc[-self.spec.horizon :]
                .values
            )
            output_col = pd.concat([output_col, output_i])

            self.forecast_output.add_category(
                category=cat, target_category_column=f"{col}_{cat}", forecast=output_i
            )

        output_col = output_col.reset_index(drop=True)

        return output_col

    def _generate_report(self):
        import datapane as dp

        sec1_text = dp.Text(
            "## Forecast Overview \nThese plots show your "
            "forecast in the context of historical data."
        )
        sec1 = utils._select_plot_list(
            lambda idx, *args: self.models[idx].plot(self.outputs[idx]),
            target_columns=self.target_columns,
        )

        sec2_text = dp.Text(f"## Forecast Broken Down by Trend Component")
        sec2 = utils._select_plot_list(
            lambda idx, *args: self.models[idx].plot_components(self.outputs[idx]),
            target_columns=self.target_columns,
        )

        sec3_text = dp.Text(f"## Forecast Parameter Plots")
        sec3 = utils._select_plot_list(
            lambda idx, *args: self.models[idx].plot_parameters(),
            target_columns=self.target_columns,
        )

        sec5_text = dp.Text(f"## Neural Prophet Model Parameters")
        model_states = []
        for i, m in enumerate(self.models):
            model_states.append(
                pd.Series(
                    m.state_dict(),
                    index=m.state_dict().keys(),
                    name=self.target_columns[i],
                )
            )
        all_model_states = pd.concat(model_states, axis=1)
        sec5 = dp.DataTable(all_model_states)

        # return [sec4_text, sec4]
        all_sections = [
            sec1_text,
            sec1,
            sec2_text,
            sec2,
            sec3_text,
            sec3,
            sec5_text,
            sec5,
        ]

        if self.spec.generate_explanations:
            try:
                # If the key is present, call the "explain_model" method
                self.explain_model(
                    datetime_col_name="ds",
                    explain_predict_fn=self._custom_predict_neuralprophet,
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
                # Do not fail the whole run due to explanations failure
                logger.warn(f"Failed to generate Explanations with error: {e}.")
                logger.debug(f"Full Traceback: {traceback.format_exc()}")

        model_description = dp.Text(
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

    def _custom_predict_neuralprophet(self, data):
        raise NotImplementedError("NeuralProphet does not yet support explanations.")
        # data_prepped = data.reset_index()
        # data_prepped['y'] = None
        # data_prepped['ds'] = pd.to_datetime(data_prepped['ds'])
        # return self.models[self.target_columns.index(self.series_id)].predict(data_prepped)["yhat1"]
