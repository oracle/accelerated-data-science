#!/usr/bin/env python
# -*- coding: utf-8; -*-
import copy

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl.operator.lowcode.forecast.operator_config import *
from ads.opctl.operator.lowcode.forecast.model.factory import ForecastOperatorModelFactory
import pandas as pd
from ads.opctl import logger
import os

if __name__ == '__main__':
    """Benchmarks for datasets."""

    try:
        data_dir = os.environ["OCI__FORECASTING_DATA_DIR"]
    except:
        raise ValueError("Please set the environment variable `OCI__FORECASTING_DATA_DIR` to the location of the "
                         "forecasting datasets")

    smape = SupportedMetrics.SMAPE
    mape = SupportedMetrics.MAPE
    rmse = SupportedMetrics.RMSE

    prophet = 'prophet'
    arima = 'arima'
    automlx = 'automlx'
    neuralprophet = 'neuralprophet'

    benchmark_metrics = [smape, mape, rmse]

    # Expected values
    cust1_numbers = {
        prophet: {smape: 30, mape: 10, rmse: 1780},
        arima: {smape: 20, mape: 2, rmse: 1500},
        automlx: {smape: 30, mape: 7, rmse: 1750},
        # neuralprophet: {smape: 29, mape: 9.5, rmse: 1760},
    }

    cust2_numbers = {
        prophet: {smape: 18, mape: 0.5, rmse: 75},
        arima: {smape: 21, mape: 0.45, rmse: 75},
        automlx: {smape: 15, mape: 0.3, rmse: 74},
        # neuralprophet: {smape: 30, mape: 10, rmse: 1780},
    }

    datasets = {
        'cust1': cust1_numbers,
        'cust2': cust2_numbers,
    }
    metrics = [SupportedMetrics.SMAPE, SupportedMetrics.MAPE, SupportedMetrics.RMSE]

    for dataset in datasets:
        for model in datasets[dataset]:
            operator_config: ForecastOperatorConfig = ForecastOperatorConfig.from_yaml(
                uri=os.path.join(data_dir, dataset, 'forecast.yaml')
            )
            operator_config.spec.model = model
            operator_config.spec.output_directory = OutputDirectory(
                url=os.path.join(operator_config.spec.output_directory.url, model)
            )

            # Training and generating the model outputs
            ForecastOperatorModelFactory.get_model(operator_config).generate_report()

            # Reading holdout errors.
            metrics_df = pd.read_csv(os.path.join(data_dir, dataset, 'output', model, 'metrics.csv')).set_index(
                'metrics')
            metrics_dict = metrics_df.mean(axis=1).to_dict()
            logger.info("{} | {} | {}".format(dataset, model, metrics_dict))
            # Actual values should be less than actual values
            for metric in benchmark_metrics:
                assert metrics_dict[metric] <= datasets[dataset][model][metric]
            logger.info("Test completed for {} and {} model".format(dataset, model))
