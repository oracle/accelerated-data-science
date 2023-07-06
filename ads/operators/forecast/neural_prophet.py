#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import logging
import os
import sys
from ads.jobs.builders.runtimes.python_runtime import PythonRuntime
import pandas as pd
from ads.operators.forecast.utils import load_data_dict, _write_data
import numpy as np

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def _get_np_metrics_dict(selected_metric):
    from torchmetrics.regression import (
        MeanAbsolutePercentageError,
        SymmetricMeanAbsolutePercentageError,
        MeanAbsoluteError,
        R2Score,
        MeanSquaredError,
    )

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


def _preprocess_prophet(data, ds_column, datetime_format):
    data["ds"] = pd.to_datetime(data[ds_column], format=datetime_format)
    return data.drop([ds_column], axis=1)


def _fit_neuralprophet_model(data, params, additional_regressors, select_metric):
    from neuralprophet import NeuralProphet

    m = NeuralProphet(**params)
    m.metrics = _get_np_metrics_dict(select_metric)
    for add_reg in additional_regressors:
        m = m.add_future_regressor(name=add_reg)
    m.fit(df=data)
    accepted_regressors_config = m.config_regressors or dict()
    return m, list(accepted_regressors_config.keys())


def operate(operator):
    from neuralprophet import NeuralProphet

    operator = load_data_dict(operator)
    full_data_dict = operator.full_data_dict

    models = []
    outputs = dict()
    outputs_legacy = []

    # Extract the Confidence Interval Width and convert to neural prophets equivalent - quantiles
    model_kwargs = operator.model_kwargs
    if operator.confidence_interval_width is None:
        quantiles = operator.model_kwargs.get("quantiles", [0.05, 0.95])
        operator.confidence_interval_width = float(quantiles[1]) - float(quantiles[0])
    else:
        boundaries = round((1 - operator.confidence_interval_width) / 2, 2)
        quantiles = [boundaries, operator.confidence_interval_width + boundaries]
    model_kwargs["quantiles"] = quantiles

    for i, (target, df) in enumerate(full_data_dict.items()):
        model_kwargs_i = model_kwargs.copy()
        # format the dataframe for this target. Dropping NA on target[df] will remove all future data
        df = _preprocess_prophet(df, operator.ds_column, operator.datetime_format)
        data_i = df[df[target].notna()]
        data_i.rename({target: "y"}, axis=1, inplace=True)

        # Assume that all columns passed in should be used as additional data
        additional_regressors = set(data_i.columns) - {"y", "ds"}
        training_data = data_i[["y", "ds"] + list(additional_regressors)]

        if operator.perform_tuning:
            import optuna
            from torch import Tensor

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

                folds = NeuralProphet(**params).crossvalidation_split_df(data_i, k=3)
                test_metrics_total_i = []
                for df_train, df_test in folds:
                    m, accepted_regressors = _fit_neuralprophet_model(
                        data=df_train,
                        params=params,
                        additional_regressors=additional_regressors,
                        select_metric=operator.selected_metric,
                    )
                    # m = NeuralProphet(**params)
                    # m.metrics = _get_np_metrics_dict(operator.selected_metric)
                    # for add_reg in additional_regressors:
                    #     m = m.add_future_regressor(name=add_reg)
                    # m.fit(df=df_train)

                    # accepted_regressors_config = m.config_regressors or dict()
                    # accepted_regressors = list(accepted_regressors_config.keys())
                    df_test = df_test[["y", "ds"] + accepted_regressors]

                    test_forecast_i = m.predict(df=df_test)
                    fold_metric_i = (
                        m.metrics[operator.selected_metric]
                        .forward(
                            Tensor(test_forecast_i["yhat1"]),
                            Tensor(test_forecast_i["y"]),
                        )
                        .item()
                    )
                    test_metrics_total_i.append(fold_metric_i)
                print(
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
            study.optimize(objective, n_trials=operator.num_tuning_trials, n_jobs=-1)

            selected_params = study.best_params
            selected_params.update(model_kwargs_i)
            model_kwargs_i = selected_params

        # Build and fit model
        model, accepted_regressors = _fit_neuralprophet_model(
            data=training_data,
            params=model_kwargs_i,
            additional_regressors=additional_regressors,
            select_metric=operator.selected_metric,
        )
        # model = NeuralProphet(**model_kwargs_i)
        # model.metrics = _get_np_metrics_dict(operator.selected_metric)
        # for add_reg in additional_regressors:
        #     model = model.add_future_regressor(name=add_reg)
        # model.fit(training_data, freq=operator.horizon["interval_unit"])

        # Determine which regressors were accepted
        # accepted_regressors_config = model.config_regressors or dict()
        # accepted_regressors = list(accepted_regressors_config.keys())
        print(f"Found the following additional data columns: {additional_regressors}")
        print(
            f"While fitting the model, some additional data may have been discarded. Only using the columns: {accepted_regressors}"
        )

        # Build future dataframe
        future = df.reset_index(drop=True)
        future["y"] = None
        future = future[["y", "ds"] + list(accepted_regressors)]

        # Forecaset model and collect outputs
        forecast = model.predict(future)
        print(f"-----------------Model {i}----------------------")
        print(forecast.tail())
        models.append(model)
        outputs[target] = forecast
        outputs_legacy.append(forecast)

    operator.models = models
    operator.outputs = outputs_legacy

    print("===========Done===========")
    outputs_merged = pd.DataFrame()

    # Merge the outputs from each model into 1 df with all outputs by target and category
    for col in operator.original_target_columns:
        output_col = pd.DataFrame()
        for cat in operator.categories:  # Note: to restrict columns, set this to [:2]
            output_i = pd.DataFrame()

            output_i[operator.ds_column] = outputs[f"{col}_{cat}"]["ds"]
            output_i[operator.target_category_column] = cat
            output_i[f"{col}_forecast"] = outputs[f"{col}_{cat}"]["yhat1"]
            output_i[f"{col}_forecast_upper"] = outputs[f"{col}_{cat}"][
                f"yhat1 {quantiles[1]*100}%"
            ]
            output_i[f"{col}_forecast_lower"] = outputs[f"{col}_{cat}"][
                f"yhat1 {quantiles[0]*100}%"
            ]
            output_col = pd.concat([output_col, output_i])
        output_col = output_col.sort_values(operator.ds_column).reset_index(drop=True)
        outputs_merged = pd.concat([outputs_merged, output_col], axis=1)

    _write_data(
        outputs_merged, operator.output_filename, "csv", operator.storage_options
    )

    # Re-merge historical datas for processing
    data_merged = pd.concat(
        [v[v[k].notna()].set_index("ds") for k, v in full_data_dict.items()], axis=1
    ).reset_index()
    return data_merged, models, outputs_legacy


def get_neuralprophet_report(self):
    import datapane as dp

    def get_select_plot_list(fn):
        return dp.Select(
            blocks=[
                dp.Plot(fn(i), label=col) for i, col in enumerate(self.target_columns)
            ]
        )

    sec1_text = dp.Text(
        f"## Forecast Overview \nThese plots show your forecast in the context of historical data."
    )  # TODO add confidence intervals
    sec1 = get_select_plot_list(lambda idx: self.models[idx].plot(self.outputs[idx]))

    sec2_text = dp.Text(f"## Forecast Broken Down by Trend Component")
    sec2 = get_select_plot_list(
        lambda idx: self.models[idx].plot_components(self.outputs[idx])
    )

    sec3_text = dp.Text(f"## Forecast Parameter Plots")
    sec3 = get_select_plot_list(lambda idx: self.models[idx].plot_parameters())

    # Auto-corr
    # sec4_text = dp.Text(f"## Auto-Correlation Plots")
    # output_series = []
    # for idx in range(len(self.target_columns)):
    #     series = pd.Series(self.outputs[idx]["yhat1"])
    #     series.index = pd.DatetimeIndex(self.outputs[idx]["ds"])
    #     output_series.append(series)
    # sec4 = get_select_plot_list(lambda idx: pd.plotting.autocorrelation_plot(output_series[idx]))

    sec5_text = dp.Text(f"## Neural Prophet Model Parameters")
    model_states = []
    for i, m in enumerate(self.models):
        model_states.append(
            pd.Series(
                m.state_dict(), index=m.state_dict().keys(), name=self.target_columns[i]
            )
        )
    all_model_states = pd.concat(model_states, axis=1)
    sec5 = dp.DataTable(all_model_states)

    # return [sec4_text, sec4]
    return [sec1_text, sec1, sec2_text, sec2, sec3_text, sec3, sec5_text, sec5]


# from neuralprophet import NeuralProphet
# import pandas as pd
# df = pd.read_csv('pypistats.csv')
# df['ds'] = df['date']
# df['y'] = df['ocifs_downloads']
# df = df[['ds', 'y']]
# m = NeuralProphet()
# metrics = m.fit(df, freq="D")
# forecast = m.predict(df)
