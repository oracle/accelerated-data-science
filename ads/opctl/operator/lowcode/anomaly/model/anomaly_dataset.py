#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ..operator_config import AnomalyOperatorConfig
from .. import utils
from ads.opctl.operator.common.utils import default_signer
from ads.opctl import logger
import pandas as pd


class AnomalyDatasets:
    def __init__(self, config: AnomalyOperatorConfig):
        """Instantiates the DataIO instance.

        Properties
        ----------
        config: AnomalyOperatorConfig
            The anomaly operator configuration.
        """
        self.original_user_data = None
        self.data = None
        self.test_data = None
        self.target_columns = None
        self._load_data(config.spec)

    def _load_data(self, spec):
        """Loads anomaly input data."""

        self.data = utils._load_data(
            filename=spec.input_data.url,
            format=spec.input_data.format,
            storage_options=default_signer(),
            columns=spec.input_data.columns,
        )
        self.original_user_data = self.data.copy()
        date_col = spec.datetime_column.name
        self.data[date_col] = pd.to_datetime(self.data[date_col])
        try:
            spec.freq = utils.get_frequency_of_datetime(self.data, spec)
        except TypeError as e:
            logger.warn(
                f"Error determining frequency: {e.args}. Setting Frequency to None"
            )
            logger.debug(f"Full traceback: {e}")
            spec.freq = None


class AnomalyOutput:
    def __init__(
        self, inliers: pd.DataFrame, outliers: pd.DataFrame, scores: pd.DataFrame
    ):
        # Timestamp as index
        self.inliers = inliers
        self.outliers = outliers
        self.scores = scores
