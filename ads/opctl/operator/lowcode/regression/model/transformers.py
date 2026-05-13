#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import annotations

import re
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from ads.opctl.operator.lowcode.common.errors import InvalidParameterError
from ads.opctl.operator.lowcode.regression.const import ColumnType


class ColumnTypeResolver:
    """Resolves regression feature columns into numeric, categorical, and date groups."""

    @classmethod
    def infer_column_types(
        cls,
        x_df: pd.DataFrame,
        feature_columns: Sequence[str],
        configured_column_types: dict | None = None,
    ):
        numeric_cols = []
        categorical_cols = []
        date_cols = []
        configured = configured_column_types or {}

        for col in feature_columns:
            if col in configured:
                configured_type = str(configured[col]).lower()
                if configured_type == ColumnType.CATEGORICAL:
                    categorical_cols.append(col)
                elif configured_type == ColumnType.DATE:
                    date_cols.append(col)
                elif configured_type == ColumnType.NUMERICAL:
                    numeric_cols.append(col)
                else:
                    raise InvalidParameterError(
                        f"Unsupported column type `{configured[col]}` for column `{col}`. "
                        f"Supported values are `{ColumnType.NUMERICAL}`, `{ColumnType.CATEGORICAL}`, and `{ColumnType.DATE}`."
                    )
                continue

            series = x_df[col]
            if cls.is_datetime_like(series, col):
                date_cols.append(col)
            elif cls.is_categorical_like(series, col):
                categorical_cols.append(col)
            elif cls.is_numeric_like(series):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        return numeric_cols, categorical_cols, date_cols

    @staticmethod
    def normalize_string_series(series: pd.Series) -> pd.Series:
        normalized = series.astype(str).str.strip()
        return normalized.replace(
            {
                "": np.nan,
                "nan": np.nan,
                "NaN": np.nan,
                "none": np.nan,
                "None": np.nan,
                "null": np.nan,
                "NULL": np.nan,
            }
        )

    @classmethod
    def is_numeric_like(cls, series: pd.Series) -> bool:
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_datetime64_any_dtype(
            series
        ):
            return False
        if pd.api.types.is_numeric_dtype(series):
            return True

        cleaned = cls.normalize_string_series(series)
        non_null = cleaned.dropna()
        if non_null.empty:
            return False

        parsed = pd.to_numeric(
            non_null.str.replace(",", "", regex=False),
            errors="coerce",
        )
        return (parsed.notna().sum() / len(non_null)) >= 0.95

    @classmethod
    def is_categorical_like(cls, series: pd.Series, column_name: str) -> bool:
        if pd.api.types.is_bool_dtype(series):
            return True
        if cls.looks_like_identifier(column_name):
            return True
        if pd.api.types.is_numeric_dtype(series):
            return cls.has_low_numeric_cardinality(series)

        cleaned = cls.normalize_string_series(series)
        non_null = cleaned.dropna()
        if non_null.empty:
            return True
        if cls.is_numeric_like(series):
            return cls.looks_like_identifier(column_name)
        return True

    @classmethod
    def is_datetime_like(cls, series: pd.Series, column_name: str = "") -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            return False

        cleaned = cls.normalize_string_series(series)
        non_null = cleaned.dropna()
        if non_null.empty:
            return False
        if cls.is_numeric_like(series):
            return False

        sample = non_null.astype(str).head(min(len(non_null), 50))
        has_date_signal = (
            sample.str.contains(r"[-/:T]|[A-Za-z]{3,}|\d{8}", regex=True).mean() >= 0.6
        )
        if not has_date_signal:
            return False

        parsed = pd.to_datetime(non_null, errors="coerce")
        parsed_ratio = parsed.notna().sum() / len(non_null)
        threshold = 0.5 if cls.looks_like_date_column(column_name) else 0.8
        return parsed_ratio >= threshold

    @staticmethod
    def has_low_numeric_cardinality(series: pd.Series) -> bool:
        clean = pd.Series(series).dropna()
        if clean.empty:
            return True

        unique_count = clean.nunique()
        row_count = len(clean)
        threshold = max(5, min(20, int(row_count * 0.05)))
        return unique_count <= threshold

    @staticmethod
    def looks_like_identifier(column_name: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(column_name).lower())
        return bool(
            re.search(
                r"(^|_)(id|key|code|zip|zipcode|postal|phone|account|customer|user|order|invoice)(_|$)",
                normalized,
            )
        )

    @staticmethod
    def looks_like_date_column(column_name: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(column_name).lower())
        return bool(
            re.search(
                r"(^|_)(date|time|timestamp|datetime)(_|$)",
                normalized,
            )
        )


class RegressionFeaturePreprocessor:
    """Plain regression preprocessor with explicit training and inference methods."""

    def __init__(
        self,
        feature_columns: Sequence[str],
        column_types: dict | None = None,
        preprocessing_enabled: bool = True,
        missing_value_imputation: bool = True,
        categorical_encoding: bool = True,
    ):
        self.feature_columns = list(feature_columns or [])
        self.column_types = column_types or {}
        self.preprocessing_enabled = preprocessing_enabled
        self.missing_value_imputation = missing_value_imputation
        self.categorical_encoding = categorical_encoding
        self.numeric_fill_values_ = {}
        self.categorical_fill_values_ = {}
        self.categorical_encoder_ = None
        self.output_feature_names_ = []

    def preprocess_for_training(self, X):
        x_df = self._to_frame(X)
        self.feature_columns_ = [col for col in self.feature_columns if col in x_df.columns]
        if not self.feature_columns_:
            self.feature_columns_ = list(x_df.columns)

        self.numeric_cols_, self.categorical_cols_, self.date_cols_ = (
            ColumnTypeResolver.infer_column_types(
                x_df=x_df,
                feature_columns=self.feature_columns_,
                configured_column_types=self.column_types,
            )
        )
        return self._preprocess_frame(x_df, is_training=True)

    def preprocess_for_prediction(self, X):
        x_df = self._to_frame(X)
        x_df = self._align_input_columns(x_df)
        return self._preprocess_frame(x_df, is_training=False)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(getattr(self, "output_feature_names_", []), dtype=object)

    def _to_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()

        columns = list(getattr(self, "feature_columns_", self.feature_columns))
        return pd.DataFrame(X, columns=columns or None)

    def _align_input_columns(self, x_df: pd.DataFrame) -> pd.DataFrame:
        aligned = x_df.copy()
        for col in getattr(self, "feature_columns_", self.feature_columns):
            if col not in aligned.columns:
                aligned[col] = None
        return aligned[list(getattr(self, "feature_columns_", self.feature_columns))]

    def _preprocess_frame(self, x_df: pd.DataFrame, is_training: bool) -> np.ndarray:
        numeric_df = self._preprocess_numeric_columns(x_df, is_training=is_training)
        categorical_df = self._preprocess_categorical_columns(x_df, is_training=is_training)
        date_df = self._preprocess_date_columns(x_df)

        parts = [df for df in [numeric_df, categorical_df, date_df] if not df.empty]
        if not parts:
            self.output_feature_names_ = []
            return np.empty((len(x_df), 0))

        processed = pd.concat(parts, axis=1)
        self.output_feature_names_ = list(processed.columns)
        return processed.to_numpy(dtype=float)

    def _preprocess_numeric_columns(self, x_df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        if not getattr(self, "numeric_cols_", None):
            return pd.DataFrame(index=x_df.index)

        transformed = {}
        for col in self.numeric_cols_:
            series = x_df[col]
            if pd.api.types.is_numeric_dtype(series):
                numeric_series = pd.to_numeric(series, errors="coerce")
            else:
                normalized = ColumnTypeResolver.normalize_string_series(series)
                normalized = normalized.str.replace(",", "", regex=False)
                numeric_series = pd.to_numeric(normalized, errors="coerce")

            if self.preprocessing_enabled and self.missing_value_imputation:
                if is_training:
                    non_null = numeric_series.dropna()
                    self.numeric_fill_values_[col] = (
                        float(non_null.median()) if not non_null.empty else 0.0
                    )
                numeric_series = numeric_series.fillna(self.numeric_fill_values_.get(col, 0.0))

            transformed[col] = numeric_series.astype(float)

        return pd.DataFrame(transformed, index=x_df.index)

    def _preprocess_categorical_columns(self, x_df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        if not getattr(self, "categorical_cols_", None):
            return pd.DataFrame(index=x_df.index)

        prepared = x_df[self.categorical_cols_].copy()
        for col in prepared.columns:
            prepared[col] = ColumnTypeResolver.normalize_string_series(prepared[col]).astype(object)

        if self.preprocessing_enabled and self.missing_value_imputation:
            for col in prepared.columns:
                if is_training:
                    non_null = prepared[col].dropna()
                    self.categorical_fill_values_[col] = (
                        str(non_null.mode(dropna=True).iloc[0])
                        if not non_null.empty
                        else "__missing__"
                    )
                prepared[col] = prepared[col].fillna(
                    self.categorical_fill_values_.get(col, "__missing__")
                )

        if self.preprocessing_enabled and self.categorical_encoding:
            if is_training:
                try:
                    self.categorical_encoder_ = OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False
                    )
                except TypeError:
                    self.categorical_encoder_ = OneHotEncoder(
                        handle_unknown="ignore", sparse=False
                    )
                self.categorical_encoder_.fit(prepared)

            encoded = self.categorical_encoder_.transform(prepared)
            columns = self.categorical_encoder_.get_feature_names_out(self.categorical_cols_)
            return pd.DataFrame(encoded, columns=columns, index=x_df.index)

        return prepared.reset_index(drop=True).astype(str).set_index(x_df.index).rename(columns=str)

    def _preprocess_date_columns(self, x_df: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "date_cols_", None):
            return pd.DataFrame(index=x_df.index)

        frames = []
        for col in self.date_cols_:
            dt_series = self._parse_datetime_series(x_df[col])
            output = pd.DataFrame(index=x_df.index)
            output[f"{col}_year"] = dt_series.dt.year.astype(float)
            output[f"{col}_month"] = dt_series.dt.month.astype(float)
            output[f"{col}_day"] = dt_series.dt.day.astype(float)
            output[f"{col}_dayofweek"] = dt_series.dt.dayofweek.astype(float)
            output[f"{col}_dayofyear"] = dt_series.dt.dayofyear.astype(float)
            frames.append(output.fillna(0.0))

        return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=x_df.index)

    @staticmethod
    def _parse_datetime_series(series: pd.Series) -> pd.Series:
        if pd.api.types.is_datetime64_any_dtype(series):
            return pd.to_datetime(series, errors="coerce")

        normalized = series.replace(r"^\s*$", np.nan, regex=True)
        parsed = normalized.apply(RegressionFeaturePreprocessor._safe_parse_datetime)
        parsed = pd.to_datetime(parsed, errors="coerce")
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_localize(None)
        return parsed

    @staticmethod
    def _safe_parse_datetime(value):
        if pd.isna(value):
            return pd.NaT
        try:
            parsed = pd.to_datetime(value, errors="coerce")
        except Exception:
            return pd.NaT
        if pd.isna(parsed):
            return pd.NaT
        if getattr(parsed, "tzinfo", None) is not None:
            try:
                return parsed.tz_localize(None)
            except TypeError:
                return parsed.tz_convert(None)
        return parsed
