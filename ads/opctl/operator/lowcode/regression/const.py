#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.extended_enum import ExtendedEnum


class SupportedModels(ExtendedEnum):
    """Supported regression models."""

    KNN = "knn"
    XGBOOST = "xgboost"
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    AUTO = "auto"


class SupportedMetrics(ExtendedEnum):
    """Supported metrics for regression model selection and reporting."""

    RMSE = "rmse"
    MAE = "mae"
    MSE = "mse"
    R2 = "r2"
    MAPE = "mape"


class ColumnType(ExtendedEnum):
    """Supported feature column types."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATE = "date"


TROUBLESHOOTING_GUIDE = "https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/ai-operators/troubleshooting.md"
