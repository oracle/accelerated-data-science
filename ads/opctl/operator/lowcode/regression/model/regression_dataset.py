#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import List

import pandas as pd

from ads.opctl.operator.lowcode.common.errors import InvalidParameterError
from ads.opctl.operator.lowcode.common.utils import load_data
from ads.opctl.operator.lowcode.regression.operator_config import (
    RegressionOperatorConfig,
)


class RegressionDatasets:
    """Loads and validates regression datasets."""

    def __init__(self, config: RegressionOperatorConfig):
        self.config = config
        self.spec = config.spec

        self.training_data = self._load(self.spec.training_data, "training_data")
        self.test_data = self._load_optional(self.spec.test_data, "test_data")
        self.prediction_data = self._load_optional(
            self.spec.prediction_data, "prediction_data"
        )

        self._validate_target(self.training_data, "training_data")

        self.feature_columns = self._resolve_feature_columns(self.training_data)
        self.prediction_passthrough_columns = []

        self._validate_columns(self.training_data, "training_data", require_target=True)
        if self.test_data is not None:
            self._validate_columns(self.test_data, "test_data", require_target=True)
        if self.prediction_data is not None:
            self.prediction_passthrough_columns = (
                self._resolve_prediction_passthrough_columns()
            )
            self._validate_columns(
                self.prediction_data, "prediction_data", require_target=False
            )
            self._validate_prediction_output_contract()

    def _load(self, data_spec, name: str) -> pd.DataFrame:
        try:
            return load_data(data_spec)
        except Exception as e:
            raise InvalidParameterError(f"Unable to load `{name}`. Error: {e}")

    def _load_optional(self, data_spec, name: str):
        if data_spec is None:
            return None
        has_inline_data = getattr(data_spec, "data", None) is not None
        has_external_source = any(
            bool(getattr(data_spec, attribute, None))
            for attribute in ("url", "sql", "table_name", "connect_args")
        )
        if not has_inline_data and not has_external_source:
            return None
        return self._load(data_spec, name)

    def _validate_target(self, data: pd.DataFrame, name: str):
        if self.spec.target_column not in data.columns:
            raise InvalidParameterError(
                f"Column `{self.spec.target_column}` is missing from `{name}`."
            )

    def _resolve_feature_columns(self, data: pd.DataFrame) -> List[str]:
        return [col for col in data.columns if col != self.spec.target_column]

    def _validate_columns(self, data: pd.DataFrame, name: str, require_target: bool):
        missing = [c for c in self.feature_columns if c not in data.columns]
        if missing:
            raise InvalidParameterError(f"Columns {missing} are missing from `{name}`.")
        if require_target and self.spec.target_column not in data.columns:
            hint = (
                " Use `prediction_data` for an unlabeled dataset."
                if name == "test_data"
                else ""
            )
            raise InvalidParameterError(
                f"Column `{self.spec.target_column}` is missing from `{name}`.{hint}"
            )

    def _validate_prediction_output_contract(self):
        """Validates passthrough columns for batch scoring."""
        data = self.prediction_data

        if data.empty:
            raise InvalidParameterError(
                "`prediction_data` must contain at least one row."
            )

        passthrough_columns = self.prediction_passthrough_columns
        if len(passthrough_columns) != len(set(passthrough_columns)):
            raise InvalidParameterError(
                "`prediction_output.passthrough_columns` contains duplicate column names."
            )
        missing_passthrough = [
            column for column in passthrough_columns if column not in data.columns
        ]
        if missing_passthrough:
            raise InvalidParameterError(
                f"Passthrough columns {missing_passthrough} are missing from "
                "`prediction_data`."
            )

    def _resolve_prediction_passthrough_columns(self) -> List[str]:
        """Returns configured passthroughs, defaulting to every input column."""
        configured_columns = self.spec.prediction_output.passthrough_columns
        if configured_columns is None:
            return list(self.prediction_data.columns)
        return list(configured_columns)
