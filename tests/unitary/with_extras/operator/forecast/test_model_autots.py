#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import unittest
from unittest.mock import patch, Mock
import pandas as pd
import datapane as dp
import autots
from ads.opctl.operator.common.utils import _build_image, _parse_input_args
from ads.opctl.operator.lowcode.forecast.model.autots import (
    AutoTSOperatorModel,
    AUTOTS_MAX_GENERATION,
    AUTOTS_MODELS_TO_VALIDATE,
)
from ads.opctl.operator.lowcode.forecast.operator_config import (
    ForecastOperatorConfig,
    ForecastOperatorSpec,
    TestData,
    DateTimeColumn,
    OutputDirectory,
)
from ads.opctl.operator.lowcode.forecast.const import SupportedMetrics


class TestAutoTSOperatorModel(unittest.TestCase):
    """Tests the base class for the forecasting models"""

    pass

    def setUp(self):
        spec = Mock(spec=ForecastOperatorSpec)
        spec.datetime_column = Mock(spec=DateTimeColumn)
        spec.datetime_column.name = "last_day_of_week"
        spec.horizon = 3
        spec.tuning = None
        spec.model_kwargs = {}
        spec.confidence_interval_width = 0.7
        self.spec = spec

        config = Mock(spec=ForecastOperatorConfig)
        config.spec = self.spec
        self.config = config

    @patch("autots.AutoTS")
    @patch("pandas.concat")
    def test_autots_parameter_passthrough(self, mock_concat, mock_autots):
        autots = AutoTSOperatorModel(self.config)
        autots.full_data_dict = {}
        autots.target_columns = []
        autots.categories = []
        autots._build_model()

        # When model_kwargs does not have anything, defaults should be sent as parameters.
        mock_autots.assert_called_once_with(
            forecast_length=self.spec.horizon,
            frequency="infer",
            prediction_interval=self.spec.confidence_interval_width,
            max_generations=AUTOTS_MAX_GENERATION,
            no_negatives=False,
            constraint=None,
            ensemble="auto",
            initial_template="General+Random",
            random_seed=2022,
            holiday_country="US",
            subset=None,
            aggfunc="first",
            na_tolerance=1,
            drop_most_recent=0,
            drop_data_older_than_periods=None,
            model_list="fast_parallel",
            transformer_list="auto",
            transformer_max_depth=6,
            models_mode="random",
            num_validations="auto",
            models_to_validate=AUTOTS_MODELS_TO_VALIDATE,
            max_per_model_class=None,
            validation_method="backwards",
            min_allowed_train_percent=0.5,
            remove_leading_zeroes=False,
            prefill_na=None,
            introduce_na=None,
            preclean=None,
            model_interrupt=True,
            generation_timeout=None,
            current_model_file=None,
            verbose=1,
            n_jobs=-1,
        )

        mock_autots.reset_mock()

        self.spec.model_kwargs = {
            "forecast_length": "forecast_length_from_model_kwargs",
            "frequency": "frequency_from_model_kwargs",
            "prediction_interval": "prediction_interval_from_model_kwargs",
            "max_generations": "max_generations_from_model_kwargs",
            "no_negatives": "no_negatives_from_model_kwargs",
            "constraint": "constraint_from_model_kwargs",
            "ensemble": "ensemble_from_model_kwargs",
            "initial_template": "initial_template_from_model_kwargs",
            "random_seed": "random_seed_from_model_kwargs",
            "holiday_country": "holiday_country_from_model_kwargs",
            "subset": "subset_from_model_kwargs",
            "aggfunc": "aggfunc_from_model_kwargs",
            "na_tolerance": "na_tolerance_from_model_kwargs",
            "drop_most_recent": "drop_most_recent_from_model_kwargs",
            "drop_data_older_than_periods": "drop_data_older_than_periods_from_model_kwargs",
            "model_list": " model_list_from_model_kwargs",
            "transformer_list": "transformer_list_from_model_kwargs",
            "transformer_max_depth": "transformer_max_depth_from_model_kwargs",
            "models_mode": "models_mode_from_model_kwargs",
            "num_validations": "num_validations_from_model_kwargs",
            "models_to_validate": "models_to_validate_from_model_kwargs",
            "max_per_model_class": "max_per_model_class_from_model_kwargs",
            "validation_method": "validation_method_from_model_kwargs",
            "min_allowed_train_percent": "min_allowed_train_percent_from_model_kwargs",
            "remove_leading_zeroes": "remove_leading_zeroes_from_model_kwargs",
            "prefill_na": "prefill_na_from_model_kwargs",
            "introduce_na": "introduce_na_from_model_kwargs",
            "preclean": "preclean_from_model_kwargs",
            "model_interrupt": "model_interrupt_from_model_kwargs",
            "generation_timeout": "generation_timeout_from_model_kwargs",
            "current_model_file": "current_model_file_from_model_kwargs",
            "verbose": "verbose_from_model_kwargs",
            "n_jobs": "n_jobs_from_model_kwargs",
        }

        autots._build_model()

        # All parameters in model_kwargs should be passed to autots
        mock_autots.assert_called_once_with(
            forecast_length=self.spec.horizon,
            frequency=self.spec.model_kwargs.get("frequency"),
            prediction_interval=self.spec.confidence_interval_width,
            max_generations=self.spec.model_kwargs.get("max_generations"),
            no_negatives=self.spec.model_kwargs.get("no_negatives"),
            constraint=self.spec.model_kwargs.get("constraint"),
            ensemble=self.spec.model_kwargs.get("ensemble"),
            initial_template=self.spec.model_kwargs.get("initial_template"),
            random_seed=self.spec.model_kwargs.get("random_seed"),
            holiday_country=self.spec.model_kwargs.get("holiday_country"),
            subset=self.spec.model_kwargs.get("subset"),
            aggfunc=self.spec.model_kwargs.get("aggfunc"),
            na_tolerance=self.spec.model_kwargs.get("na_tolerance"),
            drop_most_recent=self.spec.model_kwargs.get("drop_most_recent"),
            drop_data_older_than_periods=self.spec.model_kwargs.get(
                "drop_data_older_than_periods"
            ),
            model_list=self.spec.model_kwargs.get("model_list"),
            transformer_list=self.spec.model_kwargs.get("transformer_list"),
            transformer_max_depth=self.spec.model_kwargs.get("transformer_max_depth"),
            models_mode=self.spec.model_kwargs.get("models_mode"),
            num_validations=self.spec.model_kwargs.get("num_validations"),
            models_to_validate=self.spec.model_kwargs.get("models_to_validate"),
            max_per_model_class=self.spec.model_kwargs.get("max_per_model_class"),
            validation_method=self.spec.model_kwargs.get("validation_method"),
            min_allowed_train_percent=self.spec.model_kwargs.get(
                "min_allowed_train_percent"
            ),
            remove_leading_zeroes=self.spec.model_kwargs.get("remove_leading_zeroes"),
            prefill_na=self.spec.model_kwargs.get("prefill_na"),
            introduce_na=self.spec.model_kwargs.get("introduce_na"),
            preclean=self.spec.model_kwargs.get("preclean"),
            model_interrupt=self.spec.model_kwargs.get("model_interrupt"),
            generation_timeout=self.spec.model_kwargs.get("generation_timeout"),
            current_model_file=self.spec.model_kwargs.get("current_model_file"),
            verbose=self.spec.model_kwargs.get("verbose"),
            n_jobs=self.spec.model_kwargs.get("n_jobs"),
        )


if __name__ == "__main__":
    unittest.main()
