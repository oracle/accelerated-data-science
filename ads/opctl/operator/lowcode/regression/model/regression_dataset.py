#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import List

import pandas as pd

from ads.opctl.operator.lowcode.common.errors import InvalidParameterError
from ads.opctl.operator.lowcode.common.utils import load_data
from ads.opctl.operator.lowcode.regression.operator_config import RegressionOperatorConfig


class RegressionDatasets:
    """Loads and validates regression datasets."""

    def __init__(self, config: RegressionOperatorConfig):
        self.config = config
        self.spec = config.spec

        self.training_data = self._load(self.spec.training_data, "training_data")
        self.test_data = self._load_optional(self.spec.test_data, "test_data")

        self._validate_target(self.training_data, "training_data")

        self.feature_columns = self._resolve_feature_columns(self.training_data)

        self._validate_columns(self.training_data, "training_data", require_target=True)
        if self.test_data is not None:
            self._validate_columns(self.test_data, "test_data", require_target=False)

    def _load(self, data_spec, name: str) -> pd.DataFrame:
        try:
            return load_data(data_spec)
        except Exception as e:
            raise InvalidParameterError(f"Unable to load `{name}`. Error: {e}")

    def _load_optional(self, data_spec, name: str):
        if data_spec is None:
            return None
        if not any(
            [
                getattr(data_spec, "url", None),
                getattr(data_spec, "data", None),
                getattr(data_spec, "sql", None),
                getattr(data_spec, "table_name", None),
                getattr(data_spec, "connect_args", None),
            ]
        ):
            return None
        return self._load(data_spec, name)

    def _validate_target(self, data: pd.DataFrame, name: str):
        if self.spec.target_column not in data.columns:
            raise InvalidParameterError(
                f"Column `{self.spec.target_column}` is missing from `{name}`."
            )

    def _resolve_feature_columns(self, data: pd.DataFrame) -> List[str]:
        return [
            col
            for col in data.columns
            if col != self.spec.target_column
        ]

    def _validate_columns(self, data: pd.DataFrame, name: str, require_target: bool):
        missing = [c for c in self.feature_columns if c not in data.columns]
        if missing:
            raise InvalidParameterError(
                f"Columns {missing} are missing from `{name}`."
            )
        if require_target and self.spec.target_column not in data.columns:
            raise InvalidParameterError(
                f"Column `{self.spec.target_column}` is missing from `{name}`."
            )
