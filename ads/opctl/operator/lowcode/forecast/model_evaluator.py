# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import numpy as np
from ads.opctl import logger
from ads.opctl.operator.lowcode.common.utils import (
    find_output_dirname,
)
from .const import ForecastOutputColumns
from .model.forecast_datasets import ForecastDatasets
from .operator_config import ForecastOperatorConfig


class ModelEvaluator:
    def __init__(self, models, k=5, subsample_ratio=0.20):
        self.models = models
        self.k = k
        self.subsample_ratio = subsample_ratio

    def generate_k_fold_data(self, datasets: ForecastDatasets, date_col: str, horizon: int):
        historical_data = datasets.historical_data.data.reset_index()
        series_col = ForecastOutputColumns.SERIES
        group_counts = historical_data[series_col].value_counts()

        sample_count = max(5, int(len(group_counts) * self.subsample_ratio))
        sampled_groups = group_counts.head(sample_count)
        sampled_historical_data = historical_data[historical_data[series_col].isin(sampled_groups.index)]

        min_group = group_counts.idxmin()
        min_series_data = historical_data[historical_data[series_col] == min_group]
        unique_dates = min_series_data[date_col].unique()

        sorted_dates = np.sort(unique_dates)
        train_window_size = [len(sorted_dates) - (i + 1) * horizon for i in range(self.k)]
        valid_train_window_size = [ws for ws in train_window_size if ws >= horizon * 3]
        if len(valid_train_window_size) < self.k:
            logger.warn(f"Only ${valid_train_window_size} backtests can be created")

        cut_offs = sorted_dates[-horizon - 1:-horizon * (self.k + 1):-horizon][:len(valid_train_window_size)]
        training_datasets = [sampled_historical_data[sampled_historical_data[date_col] <= cut_off_date] for cut_off_date
                             in cut_offs]
        test_datasets = [sampled_historical_data[sampled_historical_data[date_col] > cut_offs[0]]]
        for i, current in enumerate(cut_offs[1:]):
            test_datasets.append(sampled_historical_data[(current < sampled_historical_data[date_col]) & (
                    sampled_historical_data[date_col] <= cut_offs[i])])
        return cut_offs, training_datasets, test_datasets

    def remove_none_values(self, obj):
        if isinstance(obj, dict):
            return {k: self.remove_none_values(v) for k, v in obj.items() if k is not None and v is not None}
        else:
            return obj

    def run_all_models(self, datasets: ForecastDatasets, operator_config: ForecastOperatorConfig):
        date_col = operator_config.spec.datetime_column.name
        horizon = operator_config.spec.horizon
        cut_offs, train_sets, test_sets = self.generate_k_fold_data(datasets, date_col, horizon)

        for model in self.models:
            from .model.factory import ForecastOperatorModelFactory
            for i in range(len(cut_offs)):
                backtest_historical_data = train_sets[i]
                backtest_test_data = test_sets[i]
                output_dir = find_output_dirname(operator_config.spec.output_directory)
                output_file_path = f'{output_dir}back_test/{i}'
                from pathlib import Path
                Path(output_file_path).mkdir(parents=True, exist_ok=True)
                historical_data_url = f'{output_file_path}/historical.csv'
                test_data_url = f'{output_file_path}/test.csv'
                backtest_historical_data.to_csv(historical_data_url, index=False)
                backtest_test_data.to_csv(test_data_url, index=False)
                backtest_op_config_draft = operator_config.to_dict()
                backtest_spec = backtest_op_config_draft["spec"]
                backtest_spec["historical_data"]["url"] = historical_data_url
                backtest_spec["test_data"]["url"] = test_data_url
                backtest_spec["model"] = model
                backtest_spec["output_directory"]["url"] = output_dir
                cleaned_config = self.remove_none_values(backtest_op_config_draft)
                backtest_op_cofig = ForecastOperatorConfig.from_dict(
                    obj_dict=cleaned_config)
                datasets = ForecastDatasets(backtest_op_cofig)

                ForecastOperatorModelFactory.get_model(
                    operator_config, datasets
                ).generate_report()
