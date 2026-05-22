#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from ads.opctl.operator.lowcode.forecast.const import (
    DEFAULT_AUTO_SELECT_SERIES_BASIC_MODELS,
)
from ads.opctl.operator.lowcode.forecast.model.factory import (
    ForecastOperatorModelFactory,
)


class TestForecastOperatorModelFactory:
    """Tests the factory class which contains a list of registered forecasting operator models."""

    def test_auto_select_series_basic_uses_all_models_by_default(self):
        operator_config = SimpleNamespace(
            spec=SimpleNamespace(
                model_kwargs={},
            )
        )
        datasets = object()
        series_model_selection = pd.DataFrame({"selected_model": ["prophet"]})

        with patch(
            "ads.opctl.operator.lowcode.forecast.model.factory.ModelEvaluator"
        ) as evaluator_cls:
            evaluator_cls.return_value.find_best_model_per_series.return_value = (
                series_model_selection
            )

            selected_model = ForecastOperatorModelFactory.auto_select_series_basic_model(
                datasets, operator_config
            )

        evaluator_cls.assert_called_once_with(
            DEFAULT_AUTO_SELECT_SERIES_BASIC_MODELS,
            1,
        )
        assert selected_model == "prophet"
        assert operator_config.spec.series_model_selection.equals(series_model_selection)
