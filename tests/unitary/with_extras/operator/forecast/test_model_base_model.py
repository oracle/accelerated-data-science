#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import unittest
from unittest.mock import patch, Mock
import pandas as pd
from ads.opctl.operator.common.utils import _build_image, _parse_input_args
from ads.opctl.operator.lowcode.forecast.model.prophet import ProphetOperatorModel
from ads.opctl.operator.lowcode.forecast.model.base_model import (
    ForecastOperatorBaseModel,
)
from ads.opctl.operator.lowcode.forecast.operator_config import (
    ForecastOperatorConfig,
    ForecastOperatorSpec,
    TestData,
    DateTimeColumn,
    Horizon,
)
from ads.opctl.operator.lowcode.forecast.const import SupportedMetrics


class TestForecastOperatorBaseModel(unittest.TestCase):
    """Tests the base class for the forecasting models"""

    def setUp(self):
        self.target_columns = ["Sales_Product Group 107", "Sales_Product Group 108"]
        self.target_category_columns = ["PPG_Code"]
        self.test_filename = "test.csv"
        self.outputs = [
            pd.DataFrame(
                {
                    "ds": [
                        "2020-10-31",
                        "2020-11-07",
                        "2020-11-14",
                        "2020-11-21",
                        "2020-11-28",
                    ],
                    "yhat": [
                        1569.536030,
                        1568.052261,
                        1566.568493,
                        1565.084725,
                        1563.600957,
                    ],
                }
            ),
            pd.DataFrame(
                {
                    "ds": [
                        "2020-10-31",
                        "2020-11-07",
                        "2020-11-14",
                        "2020-11-21",
                        "2020-11-28",
                    ],
                    "yhat": [
                        1284.534104,
                        1269.692458,
                        1254.850813,
                        1240.009167,
                        1225.167521,
                    ],
                }
            ),
        ]
        self.target_col = "yhat"
        self.datetime_column_name = "last_day_of_week"
        self.original_target_column = "Sales"
        self.evaluation_metrics = [
            SupportedMetrics.SMAPE,
            SupportedMetrics.MAPE,
            SupportedMetrics.RMSE,
            SupportedMetrics.R2,
            SupportedMetrics.EXPLAINED_VARIANCE,
        ]
        self.summary_metrics = [
            SupportedMetrics.MEAN_SMAPE,
            SupportedMetrics.MEDIAN_SMAPE,
            SupportedMetrics.MEAN_MAPE,
            SupportedMetrics.MEDIAN_MAPE,
            SupportedMetrics.MEAN_WMAPE,
            SupportedMetrics.MEDIAN_WMAPE,
            SupportedMetrics.MEAN_RMSE,
            SupportedMetrics.MEDIAN_RMSE,
            SupportedMetrics.MEAN_R2,
            SupportedMetrics.MEDIAN_R2,
            SupportedMetrics.MEAN_EXPLAINED_VARIANCE,
            SupportedMetrics.MEDIAN_EXPLAINED_VARIANCE,
            SupportedMetrics.ELAPSED_TIME,
        ]
        self.summary_metrics_all_targets = [
            SupportedMetrics.MEAN_SMAPE,
            SupportedMetrics.MEDIAN_SMAPE,
            SupportedMetrics.MEAN_MAPE,
            SupportedMetrics.MEDIAN_MAPE,
            SupportedMetrics.MEAN_RMSE,
            SupportedMetrics.MEDIAN_RMSE,
            SupportedMetrics.MEAN_R2,
            SupportedMetrics.MEDIAN_R2,
            SupportedMetrics.MEAN_EXPLAINED_VARIANCE,
            SupportedMetrics.MEDIAN_EXPLAINED_VARIANCE,
            SupportedMetrics.ELAPSED_TIME,
        ]

        spec = Mock(spec=ForecastOperatorSpec)
        spec.target_column = self.target_col
        spec.target_category_columns = self.target_category_columns
        spec.target_column = self.original_target_column
        spec.test_data = Mock(spec=TestData)
        spec.datetime_column = Mock(spec=DateTimeColumn)
        spec.datetime_column.name = self.datetime_column_name
        spec.datetime_column.format = None
        spec.horizon = Mock(spec=Horizon)
        spec.horizon.periods = 3
        spec.tuning = None

        config = Mock(spec=ForecastOperatorConfig)
        config.spec = spec

        self.config = config

    @patch("ads.opctl.operator.lowcode.forecast.utils._load_data")
    def test_empty_testdata_file(self, mock__load_data):
        # When test file is empty

        mock__load_data.side_effect = pd.errors.EmptyDataError()

        prophet = ProphetOperatorModel(self.config)

        total_metrics, summary_metrics, data = prophet._test_evaluate_metrics(
            target_columns=self.target_columns,
            test_filename=self.test_filename,
            outputs=self.outputs,
            target_col=self.target_col,
            elapsed_time=0,
        )

        self.assertTrue(total_metrics.empty)
        self.assertTrue(summary_metrics.empty)
        self.assertIsNone(data)

    @patch("ads.opctl.operator.lowcode.forecast.utils._load_data")
    def test_no_series_testdata_file(self, mock__load_data):
        # When test file has no series

        mock__load_data.return_value = pd.DataFrame(
            columns=["PPG_Code", "last_day_of_week", "Sales"]
        )

        prophet = ProphetOperatorModel(self.config)

        total_metrics, summary_metrics, data = prophet._test_evaluate_metrics(
            target_columns=self.target_columns,
            test_filename=self.test_filename,
            outputs=self.outputs,
            target_col=self.target_col,
            elapsed_time=0,
        )

        self.assertTrue(total_metrics.empty)
        self.assertTrue(summary_metrics.empty)
        self.assertIsNone(data)

    @patch("ads.opctl.operator.lowcode.forecast.utils._load_data")
    def test_one_missing_series_testdata_file(self, mock__load_data):
        """
        When there are NaN values for an entire series it will be loaded as zeros. And evaluation, summary metrics will be calculated with that zeros.
        When one entire series is missing in test file i.e; missing rows
        In this case evaluation metrics and summary metrics will not involve this series
        """
        mock__load_data.return_value = pd.DataFrame(
            {
                "PPG_Code": ["Product Group 107", "Product Group 107"],
                "last_day_of_week": ["2020-11-14", "2020-11-28"],
                "Sales": [1403, 6532],
            }
        )

        prophet = ProphetOperatorModel(self.config)

        total_metrics, summary_metrics, data = prophet._test_evaluate_metrics(
            target_columns=self.target_columns,
            test_filename=self.test_filename,
            outputs=self.outputs,
            target_col=self.target_col,
            elapsed_time=0,
        )

        self.assertFalse(total_metrics.empty)
        self.assertFalse(summary_metrics.empty)

        # Missing series should not be there in evaluation metrics
        self.assertEquals(total_metrics.columns.to_list(), ["Sales_Product Group 107"])

        # Since one entire series is not there, summary metrics per horizon should not calculated
        self.assertEquals(summary_metrics.index.to_list(), ["All Targets"])

        # All metrics should be present
        self.assertEquals(total_metrics.index.to_list(), self.evaluation_metrics)
        self.assertEquals(
            summary_metrics.columns.to_list(), self.summary_metrics_all_targets
        )

    @patch("ads.opctl.operator.lowcode.forecast.utils._load_data")
    def test_missing_rows_testdata_file(self, mock__load_data):
        """
        In the case where all series are present but there are missing rows in the test file
        Suppose the missing row was for a horizon in a series, if any other series has value for that horizon then this missing value will automatically come as 0 while loading data.
        So evaluation and summary metrics will be calculated by taking missing values as zeros in this case.

        When for a horizon, every series has missing row then in loaded data that horizon will not be there.
        In this case for total metrics zeros are added for the missing values in the series to calculate evaluation metrics.
        Where as summary metrics per horizon is not calculated for that horizon.
        """

        mock__load_data.return_value = pd.DataFrame(
            {
                "PPG_Code": [
                    "Product Group 107",
                    "Product Group 107",
                    "Product Group 108",
                    "Product Group 108",
                ],
                "last_day_of_week": [
                    "2020-11-14",
                    "2020-11-28",
                    "2020-11-14",
                    "2020-11-28",
                ],
                "Sales": [1403, 6532, 1647, 1414],
            }
        )

        prophet = ProphetOperatorModel(self.config)

        total_metrics, summary_metrics, data = prophet._test_evaluate_metrics(
            target_columns=self.target_columns,
            test_filename=self.test_filename,
            outputs=self.outputs,
            target_col=self.target_col,
            elapsed_time=0,
        )
        self.assertFalse(total_metrics.empty)
        self.assertFalse(summary_metrics.empty)

        # Missing horizon should not be there in summary_metrics per horizon, metrics should be there for other horizons
        self.assertEqual(
            [
                timestamp.strftime("%Y-%m-%d")
                for timestamp in summary_metrics.index.values[1:]
            ],
            ["2020-11-14", "2020-11-28"],
        )

        # Total metrics should be there for all series.
        self.assertEqual(total_metrics.columns.to_list(), self.target_columns)

        # All metrics should be present
        self.assertEquals(total_metrics.index.to_list(), self.evaluation_metrics)
        self.assertEquals(summary_metrics.columns.to_list(), self.summary_metrics)


if __name__ == "__main__":
    unittest.main()
