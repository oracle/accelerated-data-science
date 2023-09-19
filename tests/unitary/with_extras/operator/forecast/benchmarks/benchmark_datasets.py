#!/usr/bin/env python
# -*- coding: utf-8; -*-
import copy

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl.operator.lowcode.forecast.operator_config import *
from ads.opctl.operator.lowcode.forecast.model.factory import ForecastOperatorModelFactory
import pandas as pd
from ads.opctl import logger

if __name__ == '__main__':
    """Benchmarks for datasets."""

    data_dir = "oci://ads_preview_sdk@ociodscdev/Forecasting/data/"
    smape = SupportedMetrics.SMAPE
    mape = SupportedMetrics.MAPE
    rmse = SupportedMetrics.RMSE

    prophet = 'prophet'
    arima = 'arima'
    automlx = 'automlx'
    neuralprophet = 'neuralprophet'

    benchmark_metrics = [smape, mape, rmse]

    # Expected values
    ppg_sales_benchmark_numbers = {
        prophet: {smape: 30, mape: 10, rmse: 1780},
        arima: {smape: 20, mape: 2, rmse: 1500},
        automlx: {smape: 25, mape: 6, rmse: 1530},
        # neuralprophet: {smape: 29, mape: 9.5, rmse: 1760},
    }

    ttx_small_benchmark_numbers = {
        prophet: {smape: 18, mape: 0.5, rmse: 75},
        arima: {smape: 21, mape: 0.45, rmse: 75},
        automlx: {smape: 15, mape: 0.3, rmse: 74},
        # neuralprophet: {smape: 30, mape: 10, rmse: 1780},
    }

    datasets = {
        'EPM-PPG-CODE-SALES': ppg_sales_benchmark_numbers,
        'TTX-small': ttx_small_benchmark_numbers
    }
    metrics = [SupportedMetrics.SMAPE, SupportedMetrics.MAPE, SupportedMetrics.RMSE]

    for dataset in datasets:
        for model in datasets[dataset]:
            operator_config: ForecastOperatorConfig = ForecastOperatorConfig.from_yaml(
                uri='{}/{}/forecast.yaml'.format(data_dir, dataset)
            )
            operator_config.spec.model = model
            operator_config.spec.output_directory = OutputDirectory(
                url="{}/{}".format(operator_config.spec.output_directory.url, model))

            # Training and generating the model outputs
            ForecastOperatorModelFactory.get_model(operator_config).generate_report()

            # Reading holdout erros.
            metrics_df = pd.read_csv('{}/{}/output/{}/metrics.csv'.format(data_dir, dataset, model)).set_index(
                'metrics')
            metrics_dict = metrics_df.mean(axis=1).to_dict()
            logger.info("{} | {} | {}".format(dataset, model, metrics_dict))
            # Actual values should be less than actual values
            for metric in benchmark_metrics:
                assert metrics_dict[metric] <= datasets[dataset][model][metric]
            logger.info("Test completed for {} and {} model".format(dataset, model))
