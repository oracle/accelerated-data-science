#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import random
from ads.common.extended_enum import ExtendedEnumMeta
from ads.opctl.operator.lowcode.common.const import DataColumns
from merlion.models.anomaly import autoencoder, deep_point_anomaly_detector, isolation_forest, spectral_residual, windstats, windstats_monthly
from merlion.models.anomaly.change_point import bocpd
from merlion.models.forecast import prophet


class SupportedModels(str, metaclass=ExtendedEnumMeta):
    """Supported anomaly models."""

    AutoMLX = "automlx"
    AutoTS = "autots"
    Auto = "auto"
    MerilonAD = "merlion_ad"
    # TODS = "tods"

class NonTimeADSupportedModels(str, metaclass=ExtendedEnumMeta):
    """Supported non time-based anomaly detection models."""

    OneClassSVM = "oneclasssvm"
    IsolationForest = "isolationforest"
    RandomCutForest = "randomcutforest"
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


class MerlionADSubmodels(str, metaclass=ExtendedEnumMeta):
    """Supported Merlion AD sub models."""

    # point anomaly
    AUTOENCODER = "autoencoder"
    DAGMM = "dagmm"
    DBL = "dbl"
    DEEP_POINT_ANOMALY_DETECTOR = "deep_point_anomaly_detector"
    ISOLATION_FOREST = "isolation_forest"
    LOF = "lof"
    LSTM_ED = "lstm_ed"
    # RANDOM_CUT_FOREST = "random_cut_forest"
    SPECTRAL_RESIDUAL = "spectral_residual"
    STAT_RESIDUAL = "stat_residual"
    VAE = "vae"
    WINDSTATS = "windstats"
    WINDSTATS_MONTHLY = "windstats_monthly"
    ZMS = "zms"

    # forecast_based
    ARIMA = "arima"
    ETS = "ets"
    MSES = "mses"
    PROPHET = "prophet"
    SARIMA = "sarima"

    #changepoint
    BOCPD = "bocpd"


MERLIONAD_IMPORT_MODEL_MAP = {
    MerlionADSubmodels.AUTOENCODER: ".autoendcoder",
    MerlionADSubmodels.DAGMM: ".dagmm",
    MerlionADSubmodels.DBL: ".dbl",
    MerlionADSubmodels.DEEP_POINT_ANOMALY_DETECTOR: ".deep_point_anomaly_detector",
    MerlionADSubmodels.ISOLATION_FOREST: ".isolation_forest",
    MerlionADSubmodels.LOF: ".lof",
    MerlionADSubmodels.LSTM_ED: ".lstm_ed",
    # MerlionADSubmodels.RANDOM_CUT_FOREST: ".random_cut_forest",
    MerlionADSubmodels.SPECTRAL_RESIDUAL: ".spectral_residual",
    MerlionADSubmodels.STAT_RESIDUAL: ".stat_residual",
    MerlionADSubmodels.VAE: ".vae",
    MerlionADSubmodels.WINDSTATS: ".windstats",
    MerlionADSubmodels.WINDSTATS_MONTHLY: ".windstats_monthly",
    MerlionADSubmodels.ZMS: ".zms",
    MerlionADSubmodels.ARIMA: ".forecast_based.arima",
    MerlionADSubmodels.ETS: ".forecast_based.ets",
    MerlionADSubmodels.MSES: ".forecast_based.mses",
    MerlionADSubmodels.PROPHET: ".forecast_based.prophet",
    MerlionADSubmodels.SARIMA: ".forecast_based.sarima",
    MerlionADSubmodels.BOCPD: ".change_point.bocpd",
}


MERLIONAD_MODEL_MAP = {
    MerlionADSubmodels.AUTOENCODER: "AutoEncoder",
    MerlionADSubmodels.DAGMM: "DAGMM",
    MerlionADSubmodels.DBL: "DynamicBaseline",
    MerlionADSubmodels.DEEP_POINT_ANOMALY_DETECTOR: "DeepPointAnomalyDetector",
    MerlionADSubmodels.ISOLATION_FOREST: "IsolationForest",
    MerlionADSubmodels.LOF: "LOF",
    MerlionADSubmodels.LSTM_ED: "LSTMED",
    # MerlionADSubmodels.RANDOM_CUT_FOREST: "RandomCutForest",
    MerlionADSubmodels.SPECTRAL_RESIDUAL: "SpectralResidual",
    MerlionADSubmodels.STAT_RESIDUAL: "StatThreshold",
    MerlionADSubmodels.VAE: "VAE",
    MerlionADSubmodels.WINDSTATS: "WindStats",
    MerlionADSubmodels.WINDSTATS_MONTHLY: "MonthlyWindStats",
    MerlionADSubmodels.ZMS: "ZMS",
    MerlionADSubmodels.ARIMA: "ArimaDetector",
    MerlionADSubmodels.ETS: "ETSDetector",
    MerlionADSubmodels.MSES: "MSESDetector",
    MerlionADSubmodels.PROPHET: "ProphetDetector",
    MerlionADSubmodels.SARIMA: "SarimaDetector",
    MerlionADSubmodels.BOCPD: "BOCPD",
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


MERLION_DEFAULT_MODEL = "prophet"
TODS_DEFAULT_MODEL = "ocsvm"
SUBSAMPLE_THRESHOLD = 1000
