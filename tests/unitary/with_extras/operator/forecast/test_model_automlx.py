#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import unittest
from unittest.mock import patch, Mock
import pandas as pd
from ads.opctl.operator.lowcode.forecast.model.automlx import AutoMLXOperatorModel
from ads.opctl.operator.lowcode.forecast.model.forecast_datasets import ForecastDatasets

from ads.opctl.operator.lowcode.forecast.operator_config import (
    ForecastOperatorConfig,
    ForecastOperatorSpec,
    DateTimeColumn,
    InputData,
)


class TestAutoMLXOperatorModel(unittest.TestCase):
    """Tests the automlx operator model class."""

    def setUp(self):
        self.target_columns = ["Sales_Product Group 107", "Sales_Product Group 108"]
        self.target_category_columns = ["PPG_Code"]
        self.test_filename = "test.csv"
        self.primary_data = pd.DataFrame(
            {
                "PPG_Code": [
                    "Product Group 107",
                    "Product Group 107",
                    "Product Group 107",
                    "Product Group 107",
                    "Product Group 107",
                    "Product Group 107",
                    "Product Group 107",
                    "Product Group 107",
                    "Product Group 107",
                    "Product Group 107",
                    "Product Group 107",
                ],
                "last_day_of_week": [
                    "12-01-2019",
                    "19-01-2019",
                    "05-01-2019",
                    "26-01-2019",
                    "02-02-2019",
                    "09-02-2019",
                    "16-02-2019",
                    "23-02-2019",
                    "02-03-2019",
                    "09-03-2019",
                    "16-03-2019",
                ],
                "Sales": [
                    2187.0,
                    1149.0,
                    2070.0,
                    5958.0,
                    9540.0,
                    2883.0,
                    968.0,
                    1245.0,
                    1689.0,
                    1514.0,
                    1083.0,
                ],
            }
        )
        self.additional_data = pd.DataFrame(
            {
                "PPG_Code": [
                    "Product Group 107",
                    "Product Group 107",
                    "Product Group 107",
                    "Product Group 107",
                    "Product Group 107",
                    "Product Group 107",
                    "Product Group 107",
                    "Product Group 107",
                    "Product Group 107",
                    "Product Group 107",
                    "Product Group 107",
                    "Product Group 107",
                ],
                "last_day_of_week": [
                    "12-01-2019",
                    "19-01-2019",
                    "05-01-2019",
                    "26-01-2019",
                    "02-02-2019",
                    "09-02-2019",
                    "16-02-2019",
                    "23-02-2019",
                    "02-03-2019",
                    "09-03-2019",
                    "16-03-2019",
                    "23-03-2019",
                ],
                "pt1": [0, 0, 0, 3, 7, 4, 0, 0, 0, 0, 0, 0],
            }
        )

        self.target_col = "yhat"
        self.datetime_column_name = "last_day_of_week"
        self.original_target_column = "Sales"

        spec = Mock(spec=ForecastOperatorSpec)
        spec.target_column = self.target_col
        spec.target_category_columns = self.target_category_columns
        spec.target_column = self.original_target_column
        spec.datetime_column = Mock(spec=DateTimeColumn)
        spec.datetime_column.name = self.datetime_column_name
        spec.datetime_column.format = "%d-%m-%Y"
        spec.horizon = 1
        spec.tuning = None
        spec.confidence_interval_width = None
        spec.historical_data = Mock(spec="InputData")
        spec.historical_data.url = "primary.csv"
        spec.historical_data.format = None
        spec.historical_data.columns = None
        spec.additional_data = Mock(spec="InputData")
        spec.additional_data.url = "additional.csv"
        spec.additional_data.format = None
        spec.additional_data.columns = None
        spec.model_kwargs = {}
        config = Mock(spec=ForecastOperatorConfig)
        config.spec = spec

        self.config = config


if __name__ == "__main__":
    unittest.main()
