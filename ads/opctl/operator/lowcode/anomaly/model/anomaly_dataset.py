#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ..operator_config import AnomalyOperatorConfig
from .. import utils
from ads.opctl.operator.common.utils import default_signer
from ads.opctl import logger
import pandas as pd
from ads.opctl.operator.lowcode.anomaly.const import OutputColumns


class AnomalyDatasets:
    def __init__(self, config: AnomalyOperatorConfig):
        """Instantiates the DataIO instance.

        Properties
        ----------
        config: AnomalyOperatorConfig
            The anomaly operator configuration.
        """
        self.original_user_data = None
        self.data = None
        self.test_data = None
        self.target_columns = None
        self.full_data_dict = None
        self._load_data(config.spec)

    def _load_data(self, spec):
        """Loads anomaly input data."""

        self.data = utils._load_data(
            filename=spec.input_data.url,
            format=spec.input_data.format,
            storage_options=default_signer(),
            columns=spec.input_data.columns,
        )
        self.original_user_data = self.data.copy()
        date_col = spec.datetime_column.name
        self.data[date_col] = pd.to_datetime(self.data[date_col])
        try:
            spec.freq = utils.get_frequency_of_datetime(self.data, spec)
        except TypeError as e:
            logger.warn(
                f"Error determining frequency: {e.args}. Setting Frequency to None"
            )
            logger.debug(f"Full traceback: {e}")
            spec.freq = None

        if spec.target_category_columns is None:
            if spec.target_column is None:
                target_col = [
                    col
                    for col in self.data.columns
                    if col not in [spec.datetime_column.name]
                ]
                spec.target_column = target_col[0]
            self.full_data_dict = {spec.target_column: self.data}
        else:
            # Group the data by target column
            self.full_data_dict = dict(
                tuple(
                    (group, df.reset_index(drop=True))
                    for group, df in self.data.groupby(spec.target_category_columns[0])
                )
            )


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
            inliers = pd.merge(
                inliers,
                scores,
                on=self.date_column,
                how='inner')
        return inliers

    def get_outliers_by_cat(self, category: str, data: pd.DataFrame):
        anomaly = self.get_anomalies_by_cat(category)
        scores = self.get_scores_by_cat(category)
        outliers_indices = anomaly.index[anomaly[OutputColumns.ANOMALY_COL] == 1]
        outliers = data.iloc[outliers_indices]
        if scores is not None and not scores.empty:
            outliers = pd.merge(
                outliers,
                scores,
                on=self.date_column,
                how='inner')
        return outliers

    def get_inliers(self, full_data_dict):
        inliers = pd.DataFrame()

        for category in self.category_map.keys():
            inliers = pd.concat(
                [inliers, self.get_inliers_by_cat(category, full_data_dict[category])],
                axis=0,
                ignore_index=True,
            )
        return inliers

    def get_outliers(self, full_data_dict):
        outliers = pd.DataFrame()

        for category in self.category_map.keys():
            outliers = pd.concat(
                [
                    outliers,
                    self.get_outliers_by_cat(category, full_data_dict[category]),
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
