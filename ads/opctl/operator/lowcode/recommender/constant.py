#!/usr/bin/env python

# Copyright (c) 2023, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.extended_enum import ExtendedEnum

DEFAULT_SHOW_ROWS = 25
DEFAULT_REPORT_FILENAME = "report.html"


class OutputColumns(ExtendedEnum):
    """output columns for recommender operator"""

    USER_COL = "user"
    ITEM_COL = "item"
    SCORE = "score"


class SupportedMetrics(ExtendedEnum):
    """Supported recommender metrics."""

    RMSE = "RMSE"
    MAE = "MAE"


class SupportedModels(ExtendedEnum):
    """Supported recommender models."""

    SVD = "svd"
