#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import f_regression, f_classif, SelectKBest

from ads.type_discovery.typed_feature import ContinuousTypedFeature
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


class FeatureImportance:
    @staticmethod
    def _get_feature_ranking(sampled_df, target, target_type, score_func=None, k=None):
        if score_func is None:
            score_func = (
                f_regression
                if isinstance(target_type, ContinuousTypedFeature)
                else f_classif
            )
        # if there are any Nans, get the most common value from each column and replace. If entire column is Nan and not categorical, replace with 0
        if sampled_df.isnull().any().sum() > 0:
            mode_values = sampled_df.mode().iloc[0]
            for i, col in enumerate(sampled_df.columns):
                if pd.isna(mode_values[i]):
                    if sampled_df[col].dtype.name != "category":
                        mode_values[i] = 0
            sampled_df.fillna(mode_values, inplace=True)

        # label encoding of categorical features
        categorical_columns = sampled_df.select_dtypes(include=["object", "category"])
        if len(categorical_columns.columns) > 0:
            categorical_columns_encoded = categorical_columns.apply(
                preprocessing.LabelEncoder().fit_transform
            )
            sampled_df[categorical_columns.columns.values] = categorical_columns_encoded

        # convert datetimes to float
        datetime_columns = sampled_df.select_dtypes(include=["datetime64[ns]"])
        if len(datetime_columns.columns) > 0:
            sampled_df[
                datetime_columns.columns.values
            ] = datetime_columns.values.astype("float")

        timestamp_columns = sampled_df.select_dtypes(include=["datetime64[ns, UTC]"])
        if len(timestamp_columns.columns) > 0:
            sampled_df[
                timestamp_columns.columns.values
            ] = timestamp_columns.values.astype("datetime64[ns]").astype("float")

        extracter = SelectKBest(score_func=score_func, k="all")
        columns = sampled_df.columns.values
        feature_names = np.delete(columns, np.argwhere(columns == target))
        feature_scores = extracter.fit(
            sampled_df.drop(target, axis=1), sampled_df[target]
        )
        feature_ranking_df = (
            pd.DataFrame(
                sorted(
                    zip(
                        feature_scores.scores_,
                        feature_names[extracter.get_support(indices=True)],
                    ),
                    reverse=True,
                ),
                columns=["scores", "features"],
            )[["features", "scores"]]
            .fillna(0)
            .sort_values(by=["scores"], ascending=False)[:k]
        )
        return feature_ranking_df

    def __init__(self, ds, score_func=None, n=None):
        self.feature_ranking = self._get_feature_ranking(
            ds.sampled_df, ds.type_of_target(), score_func=score_func, k=n
        )

    def __repr__(self):
        return str(self.feature_ranking)

    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def show_in_notebook(self, fig_size=(10, 10)):
        """
        Shows selected features in the notebook with matplotlib.
        """

        if not utils.is_notebook():
            print("show_in_notebook called but not in notebook environment")
            return

        with seaborn.axes_style(style="whitegrid"):
            plt.figure(figsize=fig_size)
            seaborn.barplot(
                x="scores",
                y="features",
                data=self.feature_ranking,
                orient="h",
                palette="Blues_d",
            )
