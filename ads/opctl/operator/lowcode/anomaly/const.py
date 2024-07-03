#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.extended_enum import ExtendedEnumMeta
from ads.opctl.operator.lowcode.common.const import DataColumns


class SupportedModels(str, metaclass=ExtendedEnumMeta):
    """Supported anomaly models."""

    AutoMLX = "automlx"
    AutoTS = "autots"
    Auto = "auto"
    # TODS = "tods"

class NonTimeADSupportedModels(str, metaclass=ExtendedEnumMeta):
    """Supported non time-based anomaly detection models."""

    OneClassSVM = "oneclasssvm"
    IsolationForest = "isolationforest"
    # TODO : Add DBScan
    # DBScan = "dbscan"
    

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


class SupportedMetrics(str, metaclass=ExtendedEnumMeta):
    UNSUPERVISED_UNIFY95 = "unsupervised_unify95"
    UNSUPERVISED_UNIFY95_LOG_LOSS = "unsupervised_unify95_log_loss"
    UNSUPERVISED_N1_EXPERTS = "unsupervised_n-1_experts"
    RECALL = "Recall"
    PRECISION = "Precision"
    ACCURACY = "Accuracy"
    F1_SCORE = "f1_score"
    FP = "False Positive"
    FN = "False Negative"
    TP = "True Positive"
    TN = "True Negative"
    ROC_AUC = "ROC_AUC"
    PRC_AUC = "PRC_AUC"
    MCC = "MCC"
    MEAN_RECALL = "Mean Recall"
    MEAN_PRECISION = "Mean Precision"
    MEAN_ACCURACY = "Mean Accuracy"
    MEAN_F1_SCORE = "Mean f1_score"
    MEAN_ROC_AUC = "Mean ROC_AUC"
    MEAN_PRC_AUC = "Mean PRC_AUC"
    MEAN_MCC = "Mean MCC"
    MEDIAN_RECALL = "Median Recall"
    MEDIAN_PRECISION = "Median Precision"
    MEDIAN_ACCURACY = "Median Accuracy"
    MEDIAN_F1_SCORE = "Median f1_score"
    MEDIAN_ROC_AUC = "Median ROC_AUC"
    MEDIAN_PRC_AUC = "Median PRC_AUC"
    MEDIAN_MCC = "Median MCC"
    ELAPSED_TIME = "Elapsed Time"


class OutputColumns(str, metaclass=ExtendedEnumMeta):
    ANOMALY_COL = "anomaly"
    SCORE_COL = "score"
    Series = DataColumns.Series


TODS_DEFAULT_MODEL = "ocsvm"
