#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ..operator_config import AnomalyOperatorSpec
from ads.opctl.operator.lowcode.common.utils import (
    default_signer,
    merge_category_columns,
)
from ads.opctl.operator.lowcode.common.data import AbstractData
from ads.opctl.operator.lowcode.anomaly.utils import get_frequency_of_datetime
from ads.opctl import logger
import pandas as pd
from ads.opctl.operator.lowcode.anomaly.const import OutputColumns


class AnomalyData(AbstractData):
    def __init__(self, spec: AnomalyOperatorSpec):
        super().__init__(spec=spec, name="input_data")


class TestData(AbstractData):
    def __init__(self, spec: AnomalyOperatorSpec):
        super().__init__(spec=spec, name="test_data")


class ValidationData(AbstractData):
    def __init__(self, spec: AnomalyOperatorSpec):
        super().__init__(spec=spec, name="validation_data")

    def _ingest_data(self, spec):
        self.X_valid_dict = dict()
        self.y_valid_dict = dict()
        for s_id, df in self.get_dict_by_series().items():
            self.X_valid_dict[s_id] = df.drop([OutputColumns.ANOMALY_COL], axis=1)
            self.y_valid_dict[s_id] = df[OutputColumns.ANOMALY_COL]


class AnomalyDatasets:
    def __init__(self, spec: AnomalyOperatorSpec):
        """Instantiates the DataIO instance.

        Properties
        ----------
        spec: AnomalyOperatorSpec
            The anomaly operator spec.
        """
        self._data = AnomalyData(spec)
        self.data = self._data.get_data_long()
        self.full_data_dict = self._data.get_dict_by_series()
        if spec.validation_data is not None:
            self.valid_data = ValidationData(spec)
            self.X_valid_dict = self.valid_data.X_valid_dict
            self.y_valid_dict = self.valid_data.y_valid_dict

    # Returns raw data based on the series_id i.e; the merged target_category_column value
    def get_raw_data_by_cat(self, category):
        return self._data.get_raw_data_by_cat(category)


class AnomalyOutput:
    def __init__(self, date_column):
        self.category_map = dict()
        self.date_column = date_column

    def list_categories(self):
        categories = list(self.category_map.keys())
        categories.sort()
        return categories

    def add_output(self, category: str, anomalies: pd.DataFrame, scores: pd.DataFrame):
        self.category_map[category] = (anomalies, scores)

    def get_anomalies_by_cat(self, category: str):
        return self.category_map[category][0]

    def get_scores_by_cat(self, category: str):
        return self.category_map[category][1]

    def get_inliers_by_cat(self, category: str, data: pd.DataFrame):
        anomaly = self.get_anomalies_by_cat(category)
        scores = self.get_scores_by_cat(category)
        inlier_indices = anomaly.index[anomaly[OutputColumns.ANOMALY_COL] == 0]
        inliers = data.iloc[inlier_indices]
        if scores is not None and not scores.empty and self.date_column != "index":
            inliers = pd.merge(inliers, scores, on=self.date_column, how="inner")
        else:
            inliers = pd.merge(inliers, anomaly, left_index=True, right_index=True, how="inner")
        return inliers

    def get_outliers_by_cat(self, category: str, data: pd.DataFrame):
        anomaly = self.get_anomalies_by_cat(category)
        scores = self.get_scores_by_cat(category)
        outliers_indices = anomaly.index[anomaly[OutputColumns.ANOMALY_COL] == 1]
        outliers = data.iloc[outliers_indices]
        if scores is not None and not scores.empty and self.date_column != "index":
            outliers = pd.merge(outliers, scores, on=self.date_column, how="inner")
        else:
            outliers = pd.merge(outliers, anomaly, left_index=True, right_index=True, how="inner")
        return outliers

    def get_inliers(self, datasets):
        inliers = pd.DataFrame()

        for category in self.list_categories():
            inliers = pd.concat(
                [
                    inliers,
                    self.get_inliers_by_cat(category, datasets.get_raw_data_by_cat(category)),
                ],
                axis=0,
                ignore_index=True,
            )
        return inliers

    def get_outliers(self, datasets):
        outliers = pd.DataFrame()

        for category in self.list_categories():
            outliers = pd.concat(
                [
                    outliers,
                    self.get_outliers_by_cat(category, datasets.get_raw_data_by_cat(category)),
                ],
                axis=0,
                ignore_index=True,
            )
        return outliers

    def get_scores(self, target_category_columns):
        if target_category_columns is None:
            return self.get_scores_by_cat(self.list_categories()[0])

        scores = pd.DataFrame()
        for category in self.list_categories():
            score = self.get_scores_by_cat(category)
            score[target_category_columns[0]] = category
            scores = pd.concat([scores, score], axis=0, ignore_index=True)
        return scores

    def get_num_anomalies_by_cat(self, category: str):
        return (self.category_map[category][0][OutputColumns.ANOMALY_COL] == 1).sum()
