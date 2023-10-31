#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import unittest
from unittest.mock import patch, Mock
import pandas as pd
import datapane as dp
from ads.opctl.operator.common.utils import _build_image, _parse_input_args
from ads.opctl.operator.lowcode.forecast.model.prophet import ProphetOperatorModel
from ads.opctl.operator.lowcode.forecast.model.automlx import AutoMLXOperatorModel
from ads.opctl.operator.lowcode.forecast.model.base_model import (
    ForecastOperatorBaseModel,
)
from ads.opctl.operator.lowcode.forecast.operator_config import (
    ForecastOperatorConfig,
    ForecastOperatorSpec,
    TestData,
    DateTimeColumn,
    OutputDirectory,
)
from ads.opctl.operator.lowcode.forecast.const import SupportedMetrics


class TestForecastOperatorBaseModel(unittest.TestCase):
    """Tests the base class for the forecasting models"""

    pass

    def setUp(self):
        self.target_columns = ["Sales_Product Group 107", "Sales_Product Group 108"]
        self.target_category_columns = ["PPG_Code"]
        self.test_filename = "test.csv"
        self.full_data_dict = {
            "Sales_Product Group 107": pd.DataFrame(
                {
                    "ds": ["2020-10-31", "2020-11-07"],
                    "yhat": [1569.536030, 1568.052261],
                }
            ),
            "Sales_Product Group 108": pd.DataFrame(
                {
                    "ds": ["2020-10-31", "2020-11-07"],
                    "yhat": [1569.536030, 1568.052261],
                }
            ),
        }
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
        self.data = pd.DataFrame({"last_day_of_week": ["2020-10-31", "2020-11-07"]})
        self.target_col = "yhat"
        self.datetime_column_name = "last_day_of_week"
        self.original_target_column = "Sales"
        self.eval_metrics = pd.DataFrame(
            {"Sales_Product Group 107": [25.07]}, index=["sMAPE"]
        )
        spec = Mock(spec=ForecastOperatorSpec)
        spec.target_column = self.target_col
        spec.target_category_columns = self.target_category_columns
        spec.target_column = self.original_target_column
        spec.test_data = Mock(spec=TestData)
        spec.datetime_column = Mock(spec=DateTimeColumn)
        spec.datetime_column.name = self.datetime_column_name
        spec.datetime_column.format = None
        spec.horizon = 3
        spec.tuning = None
        spec.output_directory = Mock(spec=OutputDirectory)
        spec.output_directory.url = "URL"
        spec.forecast_filename = "forecast"
        spec.metrics_filename = "metrics"
        spec.test_metrics_filename = "test_metrics"
        spec.report_filename = "report"

        config = Mock(spec=ForecastOperatorConfig)
        config.spec = spec

        self.config = config

    @patch("datapane.save_report")
    @patch("ads.opctl.operator.lowcode.forecast.utils.get_forecast_plots")
    @patch("ads.opctl.operator.lowcode.forecast.utils.evaluate_metrics")
    @patch("ads.opctl.operator.lowcode.forecast.utils._write_data")
    @patch(
        "ads.opctl.operator.lowcode.forecast.model.base_model.ForecastOperatorBaseModel._test_evaluate_metrics"
    )
    @patch(
        "ads.opctl.operator.lowcode.forecast.model.base_model.ForecastOperatorBaseModel._load_data"
    )
    @patch(
        "ads.opctl.operator.lowcode.forecast.model.prophet.ProphetOperatorModel._build_model"
    )
    @patch(
        "ads.opctl.operator.lowcode.forecast.model.prophet.ProphetOperatorModel._generate_report"
    )
    @patch("ads.opctl.operator.lowcode.forecast.model.base_model.open")
    @patch("fsspec.open")
    def test_boolean_disable(
        self,
        mock_fsspec_open,
        mock_open,
        mock__generate_report,
        mock__build_model,
        mock__load_data,
        mock__test_evaluate_metrics,
        mock__write_data,
        mock_evaluate_metrics,
        mock_get_forecast_plots,
        mock_save_report,
    ):
        mock__test_evaluate_metrics.return_value = (pd.DataFrame(), None, None)
        mock__generate_report.return_value = (
            dp.Text("Description"),
            [dp.Text("Other Sections")],
            pd.to_datetime(self.data["last_day_of_week"]),
            None,
            None,
        )
        mock__load_data.return_value = None
        mock__build_model.return_value = pd.DataFrame()
        mock_evaluate_metrics.return_value = self.eval_metrics
        mock_get_forecast_plots = dp.Text("Random Text")

        self.config.spec.generate_metrics = True
        self.config.spec.generate_report = False

        prophet = ProphetOperatorModel(self.config)
        prophet.target_columns = self.target_columns
        prophet.full_data_dict = self.full_data_dict

        prophet.generate_report()

        # Metrics are generated, Report is not generated
        mock__test_evaluate_metrics.assert_called_once()
        mock_evaluate_metrics.assert_called_once()
        self.assertTrue(mock_save_report.call_count == 0)
        self.assertTrue(mock__write_data.call_count == 3)

        mock__test_evaluate_metrics.reset_mock()
        mock_evaluate_metrics.reset_mock()
        mock__write_data.reset_mock()
        mock_save_report.reset_mock()

        self.config.spec.generate_metrics = False
        self.config.spec.generate_report = True
        prophet.generate_report()

        # Metrics are generated to be included in report but not saved, Report is generated
        mock__test_evaluate_metrics.assert_called_once()
        mock_evaluate_metrics.assert_called_once()
        self.assertTrue(mock_save_report.call_count == 1)
        self.assertTrue(mock__write_data.call_count == 1)

    @patch(
        "ads.opctl.operator.lowcode.forecast.model.automlx.AutoMLXOperatorModel.explain_model"
    )
    def test_boolean_disable_explanations(self, mock_explain_model):
        self.config.spec.generate_explanations = False

        automlx = AutoMLXOperatorModel(self.config)
        automlx.outputs = self.outputs
        automlx.full_data_dict = {}
        automlx.data = self.data
        automlx.local_explanation = {"dummy": pd.DataFrame({"pt1": [1, 2, 3]})}
        automlx._generate_report()

        # Explanations are not generated
        mock_explain_model.assert_not_called()

        self.config.spec.generate_explanations = True
        automlx._generate_report()

        # Explanations are generated
        mock_explain_model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
