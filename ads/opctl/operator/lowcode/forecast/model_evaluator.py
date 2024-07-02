# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import numpy as np
import pandas as pd
from pathlib import Path

from ads.opctl import logger
from ads.opctl.operator.lowcode.common.const import DataColumns
from .model.forecast_datasets import ForecastDatasets
from .operator_config import ForecastOperatorConfig


class ModelEvaluator:
    """
    A class used to evaluate and determine the best model or framework from a given set of candidates.

    This class is responsible for comparing different models or frameworks based on specified evaluation
    metrics and returning the best-performing option.
    """
    def __init__(self, models, k=5, subsample_ratio=0.20):
        """
        Initializes the ModelEvaluator with a list of models, number of backtests and subsample ratio.

        Properties:
        ----------
        models (list): The list of model to be evaluated.
        k (int): The number of times each model is backtested to verify its performance.
        subsample_ratio (float): The proportion of the data used in the evaluation process.
        """
        self.models = models
        self.k = k
        self.subsample_ratio = subsample_ratio
        self.minimum_sample_count = 5

    def generate_cutoffs(self, unique_dates, horizon):
        sorted_dates = np.sort(unique_dates)
        train_window_size = [len(sorted_dates) - (i + 1) * horizon for i in range(self.k)]
        valid_train_window_size = [ws for ws in train_window_size if ws >= horizon * 2]
        if len(valid_train_window_size) < self.k:
            logger.warn(f"Only {valid_train_window_size} backtests can be created")
        cut_offs = sorted_dates[-horizon - 1:-horizon * (self.k + 1):-horizon][:len(valid_train_window_size)]
        return cut_offs

    def generate_k_fold_data(self, datasets: ForecastDatasets, operator_config: ForecastOperatorConfig):
        date_col = operator_config.spec.datetime_column.name
        horizon = operator_config.spec.horizon
        historical_data = datasets.historical_data.data.reset_index()
        series_col = DataColumns.Series
        group_counts = historical_data[series_col].value_counts()

        sample_count = max(self.minimum_sample_count, int(len(group_counts) * self.subsample_ratio))
        sampled_groups = group_counts.head(sample_count)
        sampled_historical_data = historical_data[historical_data[series_col].isin(sampled_groups.index)]

        min_group = group_counts.idxmin()
        min_series_data = historical_data[historical_data[series_col] == min_group]
        unique_dates = min_series_data[date_col].unique()

        cut_offs = self.generate_cutoffs(unique_dates, horizon)
        training_datasets = [sampled_historical_data[sampled_historical_data[date_col] <= cut_off_date] for cut_off_date
                             in cut_offs]
        test_datasets = [sampled_historical_data[sampled_historical_data[date_col] > cut_offs[0]]]
        for i, current in enumerate(cut_offs[1:]):
            test_datasets.append(sampled_historical_data[(current < sampled_historical_data[date_col]) & (
                    sampled_historical_data[date_col] <= cut_offs[i])])
        all_additional = datasets.additional_data.data.reset_index()
        sampled_additional_data = all_additional[all_additional[series_col].isin(sampled_groups.index)]
        max_historical_date = sampled_historical_data[date_col].max()
        additional_data = [sampled_additional_data[sampled_additional_data[date_col] <= max_historical_date]]
        for cut_off in cut_offs[:-1]:
            trimmed_additional_data = sampled_additional_data[sampled_additional_data[date_col] <= cut_off]
            additional_data.append(trimmed_additional_data)
        return cut_offs, training_datasets, additional_data, test_datasets

    def remove_none_values(self, obj):
        if isinstance(obj, dict):
            return {k: self.remove_none_values(v) for k, v in obj.items() if k is not None and v is not None}
        else:
            return obj

    def create_operator_config(self, operator_config, backtest, model, historical_data, additional_data, test_data):
        output_dir = operator_config.spec.output_directory.url
        output_file_path = f'{output_dir}/back_testing/{model}/{backtest}'
        Path(output_file_path).mkdir(parents=True, exist_ok=True)
        historical_data_url = f'{output_file_path}/historical.csv'
        additional_data_url = f'{output_file_path}/additional.csv'
        test_data_url = f'{output_file_path}/test.csv'
        historical_data.to_csv(historical_data_url, index=False)
        additional_data.to_csv(additional_data_url, index=False)
        test_data.to_csv(test_data_url, index=False)
        backtest_op_config_draft = operator_config.to_dict()
        backtest_spec = backtest_op_config_draft["spec"]
        backtest_spec["historical_data"]["url"] = historical_data_url
        if backtest_spec["additional_data"]:
            backtest_spec["additional_data"]["url"] = additional_data_url
        backtest_spec["test_data"] = {}
        backtest_spec["test_data"]["url"] = test_data_url
        backtest_spec["model"] = model
        backtest_spec['model_kwargs'] = None
        backtest_spec["output_directory"] = {"url": output_file_path}
        backtest_spec["target_category_columns"] = [DataColumns.Series]
        backtest_spec['generate_explanations'] = False
        cleaned_config = self.remove_none_values(backtest_op_config_draft)

        backtest_op_config = ForecastOperatorConfig.from_dict(
            obj_dict=cleaned_config)
        return backtest_op_config

    def run_all_models(self, datasets: ForecastDatasets, operator_config: ForecastOperatorConfig):
        cut_offs, train_sets, additional_data, test_sets = self.generate_k_fold_data(datasets, operator_config)
        metrics = {}
        for model in self.models:
            from .model.factory import ForecastOperatorModelFactory
            metrics[model] = {}
            for i in range(len(cut_offs)):
                backtest_historical_data = train_sets[i]
                backtest_additional_data = additional_data[i]
                backtest_test_data = test_sets[i]
                backtest_operator_config = self.create_operator_config(operator_config, i, model,
                                                                       backtest_historical_data,
                                                                       backtest_additional_data,
                                                                       backtest_test_data)
                datasets = ForecastDatasets(backtest_operator_config)
                ForecastOperatorModelFactory.get_model(
                    backtest_operator_config, datasets
                ).generate_report()
                test_metrics_filename = backtest_operator_config.spec.test_metrics_filename
                metrics_df = pd.read_csv(
                    f"{backtest_operator_config.spec.output_directory.url}/{test_metrics_filename}")
                metrics_df["average_across_series"] = metrics_df.drop('metrics', axis=1).mean(axis=1)
                metrics_average_dict = dict(zip(metrics_df['metrics'].str.lower(), metrics_df['average_across_series']))
                metrics[model][i] = metrics_average_dict[operator_config.spec.metric]
        return metrics

    def find_best_model(self, datasets: ForecastDatasets, operator_config: ForecastOperatorConfig):
        metrics = self.run_all_models(datasets, operator_config)
        avg_backtests_metrics = {key: sum(value.values()) / len(value.values()) for key, value in metrics.items()}
        best_model = min(avg_backtests_metrics, key=avg_backtests_metrics.get)
        logger.info(f"Among models {self.models}, {best_model} model shows better performance during backtesting.")
        backtest_stats = pd.DataFrame(metrics).rename_axis('backtest')
        backtest_stats.reset_index(inplace=True)
        output_dir = operator_config.spec.output_directory.url
        backtest_report_name = "backtest_stats.csv"
        backtest_stats.to_csv(f"{output_dir}/{backtest_report_name}", index=False)
        return best_model
