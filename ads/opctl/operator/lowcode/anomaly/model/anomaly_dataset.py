#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ..operator_config import AnomalyOperatorSpec
from ads.opctl.operator.lowcode.common.data import AbstractData
from ads.opctl.operator.lowcode.anomaly.utils import default_signer
from ads.opctl import logger
import pandas as pd
from ads.opctl.operator.lowcode.anomaly.const import OutputColumns


class AnomalyData(AbstractData):
    def __init__(self, spec: AnomalyOperatorSpec):
        super().__init__(spec=spec, name="input_data")


class TestData(AbstractData):
    def __init__(self, spec: AnomalyOperatorSpec):
        super().__init__(spec=spec, name="test_data")


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
        # self.test_data = None
        # self.target_columns = None
        self.full_data_dict = self._data.get_dict_by_series()
        # self._load_data(spec)

    # def _load_data(self, spec):
    #     """Loads anomaly input data."""
    #     try:
    #         self.data = load_data(
    #             filename=spec.input_data.url,
    #             format=spec.input_data.format,
    #             columns=spec.input_data.columns,
    #         )
    #     except InvalidParameterError as e:
    #         e.args = e.args + ("Invalid Parameter: input_data",)
    #         raise e
    #     date_col = spec.datetime_column.name
    #     self.data[date_col] = pd.to_datetime(self.data[date_col])
    #     try:
    #         spec.freq = get_frequency_of_datetime(self.data, spec)
    #     except TypeError as e:
    #         logger.warn(
    #             f"Error determining frequency: {e.args}. Setting Frequency to None"
    #         )
    #         logger.debug(f"Full traceback: {e}")
    #         spec.freq = None

    #     if spec.target_category_columns is None:
    #         if spec.target_column is None:
    #             target_col = [
    #                 col
    #                 for col in self.data.columns
    #                 if col not in [spec.datetime_column.name]
    #             ]
    #             spec.target_column = target_col[0]
    #         self.full_data_dict = {spec.target_column: self.data}
    #     else:
    #         # Merge target category columns

    #         self.data[OutputColumns.Series] = merge_category_columns(
    #             self.data, spec.target_category_columns
    #         )
    #         unique_categories = self.data[OutputColumns.Series].unique()
    #         self.full_data_dict = dict()

    #         for cat in unique_categories:
    #             data_by_cat = self.data[self.data[OutputColumns.Series] == cat].drop(
    #                 spec.target_category_columns + [OutputColumns.Series], axis=1
    #             )
    #             self.full_data_dict[cat] = data_by_cat


class AnomalyOutput:
    def __init__(self, date_column):
        self.category_map = dict()
        self.date_column = date_column

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
        if scores is not None and not scores.empty:
            inliers = pd.merge(inliers, scores, on=self.date_column, how="inner")
        return inliers

    def get_outliers_by_cat(self, category: str, data: pd.DataFrame):
        anomaly = self.get_anomalies_by_cat(category)
        scores = self.get_scores_by_cat(category)
        outliers_indices = anomaly.index[anomaly[OutputColumns.ANOMALY_COL] == 1]
        outliers = data.iloc[outliers_indices]
        if scores is not None and not scores.empty:
            outliers = pd.merge(outliers, scores, on=self.date_column, how="inner")
        return outliers

    def get_inliers(self, data):
        inliers = pd.DataFrame()

        for category in self.category_map.keys():
            inliers = pd.concat(
                [
                    inliers,
                    self.get_inliers_by_cat(
                        category,
                        data[data[OutputColumns.Series] == category]
                        .reset_index(drop=True)
                        .drop(OutputColumns.Series, axis=1),
                    ),
                ],
                axis=0,
                ignore_index=True,
            )
        return inliers

    def get_outliers(self, data):
        outliers = pd.DataFrame()

        for category in self.category_map.keys():
            outliers = pd.concat(
                [
                    outliers,
                    self.get_outliers_by_cat(
                        category,
                        data[data[OutputColumns.Series] == category]
                        .reset_index(drop=True)
                        .drop(OutputColumns.Series, axis=1),
                    ),
                ],
                axis=0,
                ignore_index=True,
            )
        return outliers

    def get_scores(self, target_category_columns):
        if target_category_columns is None:
            return self.get_scores_by_cat(list(self.category_map.keys())[0])

        scores = pd.DataFrame()
        for category in self.category_map.keys():
            score = self.get_scores_by_cat(category)
            score[target_category_columns[0]] = category
            scores = pd.concat([scores, score], axis=0, ignore_index=True)
        return scores

    def get_num_anomalies_by_cat(self, category: str):
        return (self.category_map[category][0][OutputColumns.ANOMALY_COL] == 1).sum()
