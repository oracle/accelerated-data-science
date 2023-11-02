#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd
from ..operator_config import ForecastOperatorConfig
from .. import utils
from ads.common.auth import default_signer
from .transformations import Transformations
from ads.opctl import logger


class ForecastDatasets:
    def __init__(self, config: ForecastOperatorConfig):
        """Instantiates the DataIO instance.

        Properties
        ----------
        config: ForecastOperatorConfig
            The forecast operator configuration.
        """
        self.original_user_data = None
        self.original_total_data = None
        self.original_additional_data = None
        self.full_data_dict = None
        self.target_columns = None
        self.categories = None
        # self.test_eval_metrics = None
        # self.original_target_column = spec.target_column
        self._load_data(config.spec)

    def _load_data(self, spec):
        """Loads forecasting input data."""

        raw_data = utils._load_data(
            filename=spec.historical_data.url,
            format=spec.historical_data.format,
            storage_options=default_signer(),
            columns=spec.historical_data.columns,
        )
        self.original_user_data = raw_data.copy()
        try:
            spec.freq = utils.get_frequency_of_datetime(raw_data, spec)
        except TypeError as e:
            logger.warn(
                f"Error determining frequency: {e.args}. Setting Frequency to None"
            )
            logger.debug(f"Full traceback: {e}")
            spec.freq = None
        data = Transformations(raw_data, spec).run()
        self.original_total_data = data
        additional_data = None
        if spec.additional_data is not None:
            additional_data = utils._load_data(
                filename=spec.additional_data.url,
                format=spec.additional_data.format,
                storage_options=default_signer(),
                columns=spec.additional_data.columns,
            )

            self.original_additional_data = additional_data.copy()
            self.original_total_data = pd.concat([data, additional_data], axis=1)
        (
            self.full_data_dict,
            self.target_columns,
            self.categories,
        ) = utils._build_indexed_datasets(
            data=data,
            target_column=spec.target_column,
            datetime_column=spec.datetime_column.name,
            horizon=spec.horizon,
            target_category_columns=spec.target_category_columns,
            additional_data=additional_data,
        )

