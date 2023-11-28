#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.extended_enum import ExtendedEnumMeta


class SupportedModels(str, metaclass=ExtendedEnumMeta):
    """Supported anomaly models."""

    AutoMLX = "automlx"
    AutoTS = "autots"
    TODS = "tods"


class SupportedMetrics(str, metaclass=ExtendedEnumMeta):
    UNSUPERVISED_UNIFY95 = "unsupervised_unify95"
    UNSUPERVISED_UNIFY95_LOG_LOSS = "unsupervised_unify95_log_loss"
    UNSUPERVISED_N1_EXPERTS = "unsupervised_n-1_experts"


class OutputColumns(str, metaclass=ExtendedEnumMeta):
    ANOMALY_COL = "anomaly"
