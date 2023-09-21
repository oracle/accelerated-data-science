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

from ...forecast.const import DEFAULT_TRIALS
from .. import utils
from .base_model import ForecastOperatorBaseModel


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

    def _build_model(self) -> pd.DataFrame:
        from neuralprophet import NeuralProphet

        full_data_dict = self.full_data_dict
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
                    logger.info(
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
            logger.info(
                f"Found the following additional data columns: {additional_regressors}"
            )
            logger.info(
                f"While fitting the model, some additional data may have been "
                f"discarded. Only using the columns: {accepted_regressors}"
            )

            # Build future dataframe
            future = df_clean.reset_index(drop=True)
            future["y"] = None
            future = future[["y", "ds"] + list(accepted_regressors)]

            # Forecast model and collect outputs
            forecast = model.predict(future)
            logger.info(f"-----------------Model {i}----------------------")
            logger.info(forecast.tail())
            models.append(model)
            outputs[target] = forecast
            outputs_legacy.append(forecast)

        self.models = models
        self.outputs = outputs_legacy

        logger.info("===========Done===========")
        outputs_merged = pd.DataFrame()

        # Merge the outputs from each model into 1 df with all outputs by target and category
        col = self.original_target_column
        output_col = pd.DataFrame()
        for cat in self.categories:  # Note: to restrict columns, set this to [:2]
            output_i = pd.DataFrame()

            output_i["Date"] = outputs[f"{col}_{cat}"]["ds"]
            output_i["Series"] = cat
            output_i[f"input_value"] = full_data_dict[f"{col}_{cat}"][f"{col}_{cat}"]

            output_i[f"fitted_value"] = float('nan')
            output_i[f"forecast_value"] = float('nan')
            output_i[f"p{int(quantiles[1]*100)}"] = float('nan')
            output_i[f"p{int(quantiles[0]*100)}"] = float('nan')

            output_i.iloc[:-self.spec.horizon.periods, output_i.columns.get_loc(f"fitted_value")] = outputs[f"{col}_{cat}"]["yhat1"].iloc[:-self.spec.horizon.periods].values
            output_i.iloc[-self.spec.horizon.periods:, output_i.columns.get_loc(f"forecast_value")] = outputs[f"{col}_{cat}"]["yhat1"].iloc[-self.spec.horizon.periods:].values
            output_i.iloc[-self.spec.horizon.periods:, output_i.columns.get_loc(f"p{int(quantiles[1]*100)}")] = outputs[f"{col}_{cat}"][f"yhat1 {quantiles[1]*100}%"].iloc[-self.spec.horizon.periods:].values
            output_i.iloc[-self.spec.horizon.periods:, output_i.columns.get_loc(f"p{int(quantiles[0]*100)}")] = outputs[f"{col}_{cat}"][f"yhat1 {quantiles[0]*100}%"].iloc[-self.spec.horizon.periods:].values
            output_col = pd.concat([output_col, output_i])
        # output_col = output_col.sort_values(operator.ds_column).reset_index(drop=True)
        output_col = output_col.reset_index(drop=True)
        outputs_merged = pd.concat([outputs_merged, output_col], axis=1)

        # Re-merge historical data for processing
        data_merged = pd.concat(
            [v[v[k].notna()].set_index("ds") for k, v in full_data_dict.items()], axis=1
        ).reset_index()

        self.data = data_merged
        return outputs_merged

    def _generate_report(self):
        import datapane as dp

        sec1_text = dp.Text(
            "## Forecast Overview \nThese plots show your "
            "forecast in the context of historical data."
        )  # TODO add confidence intervals
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

        model_description = dp.Text(
            "NeuralProphet is an easy to learn framework for interpretable time "
            "series forecasting. NeuralProphet is built on PyTorch and combines "
            "Neural Network and traditional time-series algorithms, inspired by "
            "Facebook Prophet and AR-Net."
        )
        other_sections = all_sections
        forecast_col_name = "yhat1"
        train_metrics = True
        ds_column_series = self.data["ds"]
        ds_forecast_col = self.outputs[0]["ds"]
        ci_col_names = None

        return (
            model_description,
            other_sections,
            forecast_col_name,
            train_metrics,
            ds_column_series,
            ds_forecast_col,
            ci_col_names,
        )
