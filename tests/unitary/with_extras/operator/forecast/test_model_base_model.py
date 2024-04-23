# #!/usr/bin/env python
# # -*- coding: utf-8; -*-

# # Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# # Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

# import unittest
# from unittest.mock import patch, Mock
# import pandas as pd
# import numpy as np
# from datetime import datetime
# from ads.opctl.operator.common.utils import _build_image, _parse_input_args
# from ads.opctl.operator.lowcode.forecast.model.prophet import ProphetOperatorModel
# from ads.opctl.operator.lowcode.forecast.model.base_model import (
#     ForecastOperatorBaseModel,
# )
# from ads.opctl.operator.lowcode.forecast.operator_config import (
#     ForecastOperatorConfig,
#     ForecastOperatorSpec,
#     TestData,
#     DateTimeColumn,
#     InputData,
# )
# from ads.opctl.operator.lowcode.forecast.const import SupportedMetrics

# import unittest
# from unittest.mock import patch, Mock
# import pandas as pd
# from ads.opctl.operator.common.utils import _build_image, _parse_input_args
# from ads.opctl.operator.lowcode.forecast.model.prophet import ProphetOperatorModel
# from ads.opctl.operator.lowcode.forecast.model.automlx import AutoMLXOperatorModel
# from ads.opctl.operator.lowcode.forecast.model.base_model import (
#     ForecastOperatorBaseModel,
# )
# from ads.opctl.operator.lowcode.forecast.operator_config import (
#     ForecastOperatorConfig,
#     ForecastOperatorSpec,
#     TestData,
#     DateTimeColumn,
#     OutputDirectory,
# )
# from ads.opctl.operator.lowcode.forecast.const import SupportedMetrics
# from ads.opctl.operator.lowcode.forecast.model.forecast_datasets import (
#     ForecastDatasets,
#     ForecastOutput,
# )


# class TestForecastOperatorBaseModel(unittest.TestCase):
#     """Tests the base class for the forecasting models"""

#     def setUp(self):
#         self.target_columns = ["Sales_Product Group 107", "Sales_Product Group 108"]
#         self.target_category_columns = ["PPG_Code"]
#         self.test_filename = "test.csv"
#         self.full_data_dict = {
#             "Sales_Product Group 107": pd.DataFrame(
#                 {
#                     "ds": ["2020-10-31", "2020-11-07"],
#                     "yhat": [1569.536030, 1568.052261],
#                 }
#             ),
#             "Sales_Product Group 108": pd.DataFrame(
#                 {
#                     "ds": ["2020-10-31", "2020-11-07"],
#                     "yhat": [1569.536030, 1568.052261],
#                 }
#             ),
#         }

#         self.data = pd.DataFrame({"last_day_of_week": ["2020-10-31", "2020-11-07"]})
#         self.target_col = "yhat"
#         self.datetime_column_name = "last_day_of_week"
#         self.original_target_column = "Sales"
#         self.eval_metrics = pd.DataFrame(
#             {"Sales_Product Group 107": [25.07]}, index=["sMAPE"]
#         )
#         self.evaluation_metrics = [
#             SupportedMetrics.SMAPE,
#             SupportedMetrics.MAPE,
#             SupportedMetrics.RMSE,
#             SupportedMetrics.R2,
#             SupportedMetrics.EXPLAINED_VARIANCE,
#         ]
#         self.summary_metrics = [
#             SupportedMetrics.MEAN_SMAPE,
#             SupportedMetrics.MEDIAN_SMAPE,
#             SupportedMetrics.MEAN_MAPE,
#             SupportedMetrics.MEDIAN_MAPE,
#             SupportedMetrics.MEAN_WMAPE,
#             SupportedMetrics.MEDIAN_WMAPE,
#             SupportedMetrics.MEAN_RMSE,
#             SupportedMetrics.MEDIAN_RMSE,
#             SupportedMetrics.MEAN_R2,
#             SupportedMetrics.MEDIAN_R2,
#             SupportedMetrics.MEAN_EXPLAINED_VARIANCE,
#             SupportedMetrics.MEDIAN_EXPLAINED_VARIANCE,
#             SupportedMetrics.ELAPSED_TIME,
#         ]
#         self.summary_metrics_all_targets = [
#             SupportedMetrics.MEAN_SMAPE,
#             SupportedMetrics.MEDIAN_SMAPE,
#             SupportedMetrics.MEAN_MAPE,
#             SupportedMetrics.MEDIAN_MAPE,
#             SupportedMetrics.MEAN_RMSE,
#             SupportedMetrics.MEDIAN_RMSE,
#             SupportedMetrics.MEAN_R2,
#             SupportedMetrics.MEDIAN_R2,
#             SupportedMetrics.MEAN_EXPLAINED_VARIANCE,
#             SupportedMetrics.MEDIAN_EXPLAINED_VARIANCE,
#             SupportedMetrics.ELAPSED_TIME,
#         ]
#         spec = Mock(spec=ForecastOperatorSpec)
#         spec.target_column = self.target_col
#         spec.target_category_columns = self.target_category_columns
#         spec.target_column = self.original_target_column
#         spec.test_data = Mock(spec=TestData)
#         spec.datetime_column = Mock(spec=DateTimeColumn)
#         spec.datetime_column.name = self.datetime_column_name
#         spec.datetime_column.format = None
#         spec.historical_data = Mock(spec="InputData")
#         spec.historical_data.url = "primary.csv"
#         spec.historical_data.format = None
#         spec.historical_data.columns = None
#         spec.horizon = 3
#         spec.tuning = None
#         spec.output_directory = Mock(spec=OutputDirectory)
#         spec.output_directory.url = "URL"
#         spec.forecast_filename = "forecast"
#         spec.metrics_filename = "metrics"
#         spec.test_metrics_filename = "test_metrics"
#         spec.report_filename = "report"

#         config = Mock(spec=ForecastOperatorConfig)
#         config.spec = spec

#         self.config = config

#         self.datasets = Mock(spec=ForecastDatasets)
#         self.datasets.original_user_data = None
#         self.datasets.original_total_data = None
#         self.datasets.original_additional_data = None
#         self.datasets.full_data_dict = None
#         self.datasets.target_columns = None
#         self.datasets.categories = None

#         def get_longest_datetime_column_mock():
#             return pd.Series(
#                 [
#                     datetime.strptime("2020-10-31", "%Y-%m-%d"),
#                     datetime.strptime("2020-11-07", "%Y-%m-%d"),
#                     datetime.strptime("2020-11-14", "%Y-%m-%d"),
#                     datetime.strptime("2020-11-21", "%Y-%m-%d"),
#                     datetime.strptime("2020-11-28", "%Y-%m-%d"),
#                 ]
#             )

#         self.datasets.get_longest_datetime_column.side_effect = (
#             get_longest_datetime_column_mock
#         )

#         self.output = ForecastOutput(confidence_interval_width=0.7, horizon=3, target_column=self.original_target_column, dt_column=self.datetime_column_name)
#         # self.output.add_series(
#         #     "Product Group 107",
#         #     "Sales_Product Group 107",
#         #     pd.DataFrame(
#         #         {
#         #             "Date": [
#         #                 datetime.strptime("2020-10-31", "%Y-%m-%d"),
#         #                 datetime.strptime("2020-11-07", "%Y-%m-%d"),
#         #                 datetime.strptime("2020-11-14", "%Y-%m-%d"),
#         #                 datetime.strptime("2020-11-21", "%Y-%m-%d"),
#         #                 datetime.strptime("2020-11-28", "%Y-%m-%d"),
#         #             ],
#         #             "Series": [
#         #                 "Product Group 107",
#         #                 "Product Group 107",
#         #                 "Product Group 107",
#         #                 "Product Group 107",
#         #                 "Product Group 107",
#         #             ],
#         #             "input_value": [1569.536030, 1568.052261, np.nan, np.nan, np.nan],
#         #             "fitted_value": [1569.536030, 1568.052261, np.nan, np.nan, np.nan],
#         #             "forecast_value": [
#         #                 np.nan,
#         #                 np.nan,
#         #                 1566.568493,
#         #                 1565.084725,
#         #                 1563.600957,
#         #             ],
#         #             "upper_bound": [
#         #                 np.nan,
#         #                 np.nan,
#         #                 1566.568493,
#         #                 1565.084725,
#         #                 1563.600957,
#         #             ],
#         #             "lower_bound": [
#         #                 np.nan,
#         #                 np.nan,
#         #                 1566.568493,
#         #                 1565.084725,
#         #                 1563.600957,
#         #             ],
#         #         }
#         #     ),
#         # )
#         # self.output.add_category(
#         #     "Product Group 108",
#         #     "Sales_Product Group 108",
#         #     pd.DataFrame(
#         #         {
#         #             "Date": [
#         #                 datetime.strptime("2020-10-31", "%Y-%m-%d"),
#         #                 datetime.strptime("2020-11-07", "%Y-%m-%d"),
#         #                 datetime.strptime("2020-11-14", "%Y-%m-%d"),
#         #                 datetime.strptime("2020-11-21", "%Y-%m-%d"),
#         #                 datetime.strptime("2020-11-28", "%Y-%m-%d"),
#         #             ],
#         #             "Series": [
#         #                 "Product Group 108",
#         #                 "Product Group 107",
#         #                 "Product Group 108",
#         #                 "Product Group 108",
#         #                 "Product Group 108",
#         #             ],
#         #             "input_value": [1569.536030, 1568.052261, np.nan, np.nan, np.nan],
#         #             "fitted_value": [1569.536030, 1568.052261, np.nan, np.nan, np.nan],
#         #             "forecast_value": [
#         #                 np.nan,
#         #                 np.nan,
#         #                 1254.850813,
#         #                 1240.009167,
#         #                 1225.167521,
#         #             ],
#         #             "upper_bound": [
#         #                 np.nan,
#         #                 np.nan,
#         #                 1254.850813,
#         #                 1240.009167,
#         #                 1225.167521,
#         #             ],
#         #             "lower_bound": [
#         #                 np.nan,
#         #                 np.nan,
#         #                 1254.850813,
#         #                 1240.009167,
#         #                 1225.167521,
#         #             ],
#         #         }
#         #     ),
#         # )

#     @patch("ads.opctl.operator.lowcode.forecast.utils._load_data")
#     def test_empty_testdata_file(self, mock__load_data):
#         # When test file is empty

#         mock__load_data.side_effect = pd.errors.EmptyDataError()
#         prophet = ProphetOperatorModel(self.config, self.datasets)
#         total_metrics, summary_metrics, data = prophet._test_evaluate_metrics(
#             target_columns=self.target_columns,
#             test_filename=self.test_filename,
#             output=self.output,
#             target_col=self.target_col,
#             elapsed_time=0,
#         )

#         self.assertTrue(total_metrics.empty)
#         self.assertTrue(summary_metrics.empty)
#         self.assertIsNone(data)

#     @patch("ads.opctl.operator.lowcode.forecast.utils._load_data")
#     def test_no_series_testdata_file(self, mock__load_data):
#         # When test file has no series

#         mock__load_data.return_value = pd.DataFrame(
#             columns=["PPG_Code", "last_day_of_week", "Sales"]
#         )

#         prophet = ProphetOperatorModel(self.config, self.datasets)
#         prophet.forecast_output = self.output
#         total_metrics, summary_metrics, data = prophet._test_evaluate_metrics(
#             target_columns=self.target_columns,
#             test_filename=self.test_filename,
#             output=self.output,
#             target_col=self.target_col,
#             elapsed_time=0,
#         )

#         self.assertTrue(total_metrics.empty)
#         self.assertTrue(summary_metrics.empty)
#         self.assertIsNone(data)

#     @patch("ads.opctl.operator.lowcode.forecast.utils._load_data")
#     def test_one_missing_series_testdata_file(self, mock__load_data):
#         """
#         When there are NaN values for an entire series it will be loaded as zeros. And evaluation, summary metrics will be calculated with that zeros.
#         When one entire series is missing in test file i.e; missing rows
#         In this case evaluation metrics and summary metrics will not involve this series
#         """
#         mock__load_data.return_value = pd.DataFrame(
#             {
#                 "PPG_Code": ["Product Group 107", "Product Group 107"],
#                 "last_day_of_week": ["2020-11-14", "2020-11-28"],
#                 "Sales": [1403, 6532],
#             }
#         )

#         prophet = ProphetOperatorModel(self.config, self.datasets)
#         prophet.forecast_output = self.output
#         total_metrics, summary_metrics, data = prophet._test_evaluate_metrics(
#             target_columns=self.target_columns,
#             test_filename=self.test_filename,
#             output=self.output,
#             target_col=self.target_col,
#             elapsed_time=0,
#         )

#         self.assertFalse(total_metrics.empty)
#         self.assertFalse(summary_metrics.empty)

#         # Missing series should not be there in evaluation metrics
#         self.assertEquals(total_metrics.columns.to_list(), ["Sales_Product Group 107"])

#         # one entire series is not there, summary metrics per horizon will be calculated and all horizons should be there
#         self.assertEqual(
#             [
#                 timestamp.strftime("%Y-%m-%d")
#                 for timestamp in summary_metrics.index.values[1:]
#             ],
#             ["2020-11-14", "2020-11-28"],
#         )

#         # All metrics should be present
#         self.assertEquals(total_metrics.index.to_list(), self.evaluation_metrics)
#         self.assertEquals(summary_metrics.columns.to_list(), self.summary_metrics)

#     @patch("ads.opctl.operator.lowcode.forecast.utils._load_data")
#     def test_missing_rows_testdata_file(self, mock__load_data):
#         """
#         In the case where all series are present but there are missing rows in the test file
#         Suppose the missing row was for a horizon in a series, if any other series has value for that horizon then this missing value will automatically come as 0 while loading data.
#         So evaluation and summary metrics will be calculated by taking missing values as zeros in this case.

#         When for a horizon, every series has missing row then in loaded data that horizon will not be there.
#         In this case for total metrics zeros are added for the missing values in the series to calculate evaluation metrics.
#         Where as summary metrics per horizon is not calculated for that horizon.
#         """

#         mock__load_data.return_value = pd.DataFrame(
#             {
#                 "PPG_Code": [
#                     "Product Group 107",
#                     "Product Group 107",
#                     "Product Group 108",
#                     "Product Group 108",
#                 ],
#                 "last_day_of_week": [
#                     "2020-11-14",
#                     "2020-11-28",
#                     "2020-11-14",
#                     "2020-11-28",
#                 ],
#                 "Sales": [1403, 6532, 1647, 1414],
#             }
#         )

#         prophet = ProphetOperatorModel(self.config, self.datasets)
#         prophet.forecast_output = self.output
#         total_metrics, summary_metrics, data = prophet._test_evaluate_metrics(
#             target_columns=self.target_columns,
#             test_filename=self.test_filename,
#             output=self.output,
#             target_col=self.target_col,
#             elapsed_time=0,
#         )
#         self.assertFalse(total_metrics.empty)
#         self.assertFalse(summary_metrics.empty)

#         # Missing horizon should not be there in summary_metrics per horizon, metrics should be there for other horizons
#         self.assertEqual(
#             [
#                 timestamp.strftime("%Y-%m-%d")
#                 for timestamp in summary_metrics.index.values[1:]
#             ],
#             ["2020-11-14", "2020-11-28"],
#         )

#         # Total metrics should be there for all series.
#         self.assertEqual(total_metrics.columns.to_list(), self.target_columns)

#         # All metrics should be present
#         self.assertEquals(total_metrics.index.to_list(), self.evaluation_metrics)
#         self.assertEquals(summary_metrics.columns.to_list(), self.summary_metrics)

#     @patch("ads.opctl.operator.lowcode.forecast.utils.get_forecast_plots")
#     @patch("ads.opctl.operator.lowcode.forecast.utils.evaluate_train_metrics")
#     @patch("ads.opctl.operator.lowcode.forecast.utils._write_data")
#     @patch(
#         "ads.opctl.operator.lowcode.forecast.model.base_model.ForecastOperatorBaseModel._test_evaluate_metrics"
#     )
#     @patch(
#         "ads.opctl.operator.lowcode.forecast.model.prophet.ProphetOperatorModel._build_model"
#     )
#     @patch(
#         "ads.opctl.operator.lowcode.forecast.model.prophet.ProphetOperatorModel._generate_report"
#     )
#     @patch("ads.opctl.operator.lowcode.forecast.model.base_model.open")
#     @patch("fsspec.open")
#     def test_boolean_disable(
#         self,
#         mock_fsspec_open,
#         mock_open,
#         mock__generate_report,
#         mock__build_model,
#         mock__test_evaluate_metrics,
#         mock__write_data,
#         mock_evaluate_train_metrics,
#         mock_get_forecast_plots,
#         mock_save_report,
#     ):
#         mock__test_evaluate_metrics.return_value = (pd.DataFrame(), None, None)
#         mock__generate_report.return_value = (
#             dp.Text("Description"),
#             [dp.Text("Other Sections")],
#         )
#         mock__build_model.return_value = pd.DataFrame()
#         mock_evaluate_train_metrics.return_value = self.eval_metrics
#         mock_get_forecast_plots = dp.Text("Random Text")

#         self.config.spec.generate_metrics = True
#         self.config.spec.generate_report = False

#         prophet = ProphetOperatorModel(self.config, self.datasets)
#         prophet.target_columns = self.target_columns
#         prophet.full_data_dict = self.full_data_dict
#         prophet.forecast_output = self.output

#         prophet.generate_report()

#         # Metrics are generated, Report is not generated
#         mock__test_evaluate_metrics.assert_called_once()
#         mock_evaluate_train_metrics.assert_called_once()
#         self.assertTrue(mock_save_report.call_count == 0)
#         self.assertTrue(mock__write_data.call_count == 3)

#         mock__test_evaluate_metrics.reset_mock()
#         mock_evaluate_train_metrics.reset_mock()
#         mock__write_data.reset_mock()
#         mock_save_report.reset_mock()

#         self.config.spec.generate_metrics = False
#         self.config.spec.generate_report = True
#         prophet.generate_report()

#         # Metrics are generated to be included in report but not saved, Report is generated
#         mock__test_evaluate_metrics.assert_called_once()
#         mock_evaluate_train_metrics.assert_called_once()
#         self.assertTrue(mock_save_report.call_count == 1)
#         self.assertTrue(mock__write_data.call_count == 1)

#     @patch(
#         "ads.opctl.operator.lowcode.forecast.model.automlx.AutoMLXOperatorModel.explain_model"
#     )
#     def test_boolean_disable_explanations(self, mock_explain_model):
#         self.config.spec.generate_explanations = False

#         automlx = AutoMLXOperatorModel(self.config, self.datasets)
#         automlx.output = self.output
#         automlx.full_data_dict = {}
#         automlx.data = self.data
#         automlx.local_explanation = {"dummy": pd.DataFrame({"pt1": [1, 2, 3]})}
#         automlx._generate_report()

#         # Explanations are not generated
#         mock_explain_model.assert_not_called()

#         self.config.spec.generate_explanations = True
#         automlx._generate_report()

#         # Explanations are generated
#         mock_explain_model.assert_called_once()


# if __name__ == "__main__":
#     unittest.main()
