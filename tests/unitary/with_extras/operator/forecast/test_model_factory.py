#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from ads.opctl.operator.lowcode.forecast.const import (
    AUTO_SELECT_SERIES,
    AutoSelectSeriesSelectionStrategy,
    DEFAULT_AUTO_SELECT_SERIES_BACKTESTING_MODELS,
)
from ads.opctl.operator.lowcode.forecast.model.factory import (
    ForecastOperatorModelFactory,
)


class TestForecastOperatorModelFactory:
    """Tests the factory class which contains a list of registered forecasting operator models."""

    def test_auto_select_series_backtesting_uses_all_models_by_default(self):
        operator_config = SimpleNamespace(
            spec=SimpleNamespace(
                model=AUTO_SELECT_SERIES,
                model_kwargs={
                    "selection_strategy": AutoSelectSeriesSelectionStrategy.BACKTESTING
                },
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

            selected_model = (
                ForecastOperatorModelFactory.auto_select_series_backtesting_model(
                    datasets, operator_config
                )
            )

        evaluator_cls.assert_called_once_with(
            DEFAULT_AUTO_SELECT_SERIES_BACKTESTING_MODELS,
            5,
        )
        assert selected_model == "prophet"
        assert operator_config.spec.series_model_selection.equals(series_model_selection)

    def test_get_auto_select_series_selection_strategy_defaults_to_meta_learning(
            self,
    ):
        operator_config = SimpleNamespace(
            spec=SimpleNamespace(model=AUTO_SELECT_SERIES, model_kwargs={})
        )

        strategy = (
            ForecastOperatorModelFactory.get_auto_select_series_selection_strategy(
                operator_config
            )
        )

        assert strategy == AutoSelectSeriesSelectionStrategy.META_LEARNING

    def test_get_auto_select_series_selection_strategy_uses_backtesting_when_configured(
            self,
    ):
        operator_config = SimpleNamespace(
            spec=SimpleNamespace(
                model=AUTO_SELECT_SERIES,
                model_kwargs={
                    "selection_strategy": AutoSelectSeriesSelectionStrategy.BACKTESTING
                },
            )
        )

        strategy = (
            ForecastOperatorModelFactory.get_auto_select_series_selection_strategy(
                operator_config
            )
        )

        assert strategy == AutoSelectSeriesSelectionStrategy.BACKTESTING

    def test_get_auto_select_series_selection_strategy_defaults_to_meta_learning_for_non_backtesting_values(
            self,
    ):
        operator_config = SimpleNamespace(
            spec=SimpleNamespace(
                model=AUTO_SELECT_SERIES,
                model_kwargs={"selection_strategy": "backtest"},
            )
        )

        strategy = (
            ForecastOperatorModelFactory.get_auto_select_series_selection_strategy(
                operator_config
            )
        )

        assert strategy == AutoSelectSeriesSelectionStrategy.META_LEARNING
