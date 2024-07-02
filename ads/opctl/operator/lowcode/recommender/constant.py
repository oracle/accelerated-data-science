#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.extended_enum import ExtendedEnumMeta

DEFAULT_SHOW_ROWS = 25
DEFAULT_REPORT_FILENAME = "report.html"

class OutputColumns(str, metaclass=ExtendedEnumMeta):
    """output columns for recommender operator"""
    USER_COL = "user"
    ITEM_COL = "item"
    SCORE = "score"

class SupportedMetrics(str, metaclass=ExtendedEnumMeta):
    """Supported recommender metrics."""
    RMSE = "RMSE"
    MAE = "MAE"

class SupportedModels(str, metaclass=ExtendedEnumMeta):
    """Supported recommender models."""
    SVD = "svd"