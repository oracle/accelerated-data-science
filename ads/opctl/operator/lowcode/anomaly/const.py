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


class TODSSubModels(str, metaclass=ExtendedEnumMeta):
    """Supported TODS sub models."""

    OCSVM = "ocsvm"
    DeepLog = "deeplog"
    Telemanom = "telemanom"
    IsolationForest = "isolationforest"
    LSTMODetector = "lstmodetector"
    KNN = "knn"


TODS_IMPORT_MODEL_MAP = {
    TODSSubModels.OCSVM: ".OCSVM_skinterface",
    TODSSubModels.DeepLog: ".DeepLog_skinterface",
    TODSSubModels.Telemanom: ".Telemanom_skinterface",
    TODSSubModels.IsolationForest: ".IsolationForest_skinterface",
    TODSSubModels.LSTMODetector: ".LSTMODetector_skinterface",
    TODSSubModels.KNN: ".KNN_skinterface",
}

TODS_MODEL_MAP = {
    TODSSubModels.OCSVM: "OCSVMSKI",
    TODSSubModels.DeepLog: "DeepLogSKI",
    TODSSubModels.Telemanom: "TelemanomSKI",
    TODSSubModels.IsolationForest: "IsolationForestSKI",
    TODSSubModels.LSTMODetector: "LSTMODetectorSKI",
    TODSSubModels.KNN: "KNNSKI",
}

TODS_DEFAULT_MODEL = "ocsvm"