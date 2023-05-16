#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import print_function, absolute_import

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from ads.common import utils
from ads.dataset import logger
from ads.dataset.helper import down_sample, up_sample, get_fill_val
from ads.dataset.progress import DummyProgressBar
from ads.dataset.recommendation import Recommendation
from ads.type_discovery.typed_feature import (
    ContinuousTypedFeature,
    DiscreteTypedFeature,
    OrdinalTypedFeature,
    CategoricalTypedFeature,
)


class RecommendationTransformer(TransformerMixin):
    def __init__(
        self,
        feature_metadata=None,
        correlation=None,
        target=None,
        is_balanced=False,
        target_type=None,
        feature_ranking=None,
        len=0,
        fix_imbalance=True,
        auto_transform=True,
        correlation_threshold=0.7,
    ):
        self.feature_metadata_ = feature_metadata
        self.correlation_ = correlation
        self.target_ = target
        self.target_type_ = target_type
        self.feature_ranking_ = feature_ranking
        self.fill_nan_dict_ = {}
        self.fill_na_target_ = None
        self.drop_columns = []
        self.reco_dict_ = None
        # self.combine_columns = []
        self.actions_performed_ = []
        self.is_balanced = is_balanced
        self.ds_len = len
        self.fix_imbalance = fix_imbalance
        self.balancing_strategy = None
        self.auto_transform = auto_transform
        self.correlation_threshold = correlation_threshold

    def __repr__(self):
        if len(self.actions_performed_) == 0:
            return "No recommendations suggested"
        return "\n".join(self.actions_performed_)

    @staticmethod
    def _build_recommendation(
        recommendations,
        recommendation_type,
        column_names,
        message,
        actions,
        recommended_action=None,
    ):
        if len(column_names) != 0:
            if recommendation_type in ["constant_column"]:
                recommendations[recommendation_type] = column_names
            else:
                if recommendation_type not in recommendations:
                    recommendations[recommendation_type] = {}
                for column in column_names:
                    if column not in recommendations:
                        recommendations[recommendation_type][column] = {}
                    recommendations[recommendation_type][column]["Message"] = message
                    recommendations[recommendation_type][column]["Action"] = actions
                    recommendations[recommendation_type][column]["Selected Action"] = (
                        recommended_action
                        if recommended_action is not None
                        else actions[0]
                    )

    def _get_recommendations(self, df):
        recommendations = {}
        # constant columns
        constant_columns = df.columns.values[df.apply(pd.Series.nunique) == 1]
        if self.target_ in constant_columns:
            raise ValueError(
                "Unable to continue with transformation. Target column is constant. Build the dataset "
                "by choosing a different target."
            )
        self._build_recommendation(
            recommendations,
            "constant_column",
            constant_columns,
            "Constant Column",
            ["Drop"],
        )

        # primary key
        for column in df.columns.values[
            df.apply(lambda x: x.nunique() / len(x) > 0.99)
        ]:
            # exclude columns of dtype object from primary key check as they could be columns like zipcode, credit card,
            # etc,. which are mostly unique but carries useful information
            if (
                (
                    "constant_column" not in recommendations
                    or column not in recommendations["constant_column"]
                )
                and column != self.target_
                and df[column].dtype.name.startswith("int")
            ):
                self._build_recommendation(
                    recommendations,
                    "primary_key",
                    [column],
                    "Contains mostly unique values({0:.2%})".format(
                        df[column].nunique() / len(df[column])
                    ),
                    ["Drop", "Do nothing"],
                    "Drop",
                )

        self.feature_metadata_[self.target_] = self.target_type_

        for column in df.columns.values[df.isnull().any()]:

            # filter out columns that were discovered as constant or primary key columns in the previous step,
            # as they would get dropped before imputation
            if (
                "constant_column" not in recommendations
                or column not in recommendations["constant_column"]
            ) and (
                "primary_key" not in recommendations
                or column not in recommendations["primary_key"]
            ):
                null_counts = df[column].isnull().sum()
                null_ratio = null_counts / len(df[column])
                self._get_na_action(recommendations, column, null_counts, null_ratio)

        if self.correlation_ is not None:
            #  highly correlated features
            corr_features = []
            if not isinstance(self.correlation_, list):
                self.correlation_ = [self.correlation_]
            for corr in self.correlation_:
                high_corr_var = np.where(corr > self.correlation_threshold)
                corr_features.extend(
                    [
                        (corr.index[x], corr.columns[y], corr.iat[x, y])
                        for x, y in zip(*high_corr_var)
                        if x != y and x < y
                    ]
                )
            if len(corr_features) > 0:
                # Apply all recommendations so far
                self.reco_dict_ = recommendations
                df = self._transform(df)

                for corr_feature in corr_features:
                    if (
                        self.target_ not in corr_feature
                        and corr_feature[0] in df.columns.values
                        and corr_feature[1] in df.columns.values
                        and corr_feature[0] != corr_feature[1]
                    ):
                        feature1_rank_and_score = self.feature_ranking_[
                            self.feature_ranking_["features"] == corr_feature[0]
                        ]["scores"]
                        rank1, score1 = (
                            feature1_rank_and_score.index[0],
                            feature1_rank_and_score.values[0],
                        )
                        feature2_rank_and_score = self.feature_ranking_[
                            self.feature_ranking_["features"] == corr_feature[1]
                        ]["scores"]
                        rank2, score2 = (
                            feature2_rank_and_score.index[0],
                            feature2_rank_and_score.values[0],
                        )
                        # if any of the features is top ranked or ranked similar, combine
                        # if ((len(df.columns) - rank1 + 1) / len(df.columns) > 0.7 or \
                        #         (len(df.columns) - rank2 + 1) / len(df.columns) > 0.7):
                        #     selected_action = "Combine with " + corr_feature[1]

                        # suggest dropping the column with lesser importance
                        if score1 > score2:
                            selected_action = "Drop " + corr_feature[1]
                        else:
                            selected_action = "Drop " + corr_feature[0]
                        # check if the corr_feature isn't target
                        self._build_recommendation(
                            recommendations,
                            "strong_correlation",
                            [corr_feature[0]],
                            "Strongly correlated with "
                            + corr_feature[1]
                            + "({0:.2%}.".format(corr_feature[2])
                            + ")",
                            [
                                "Drop " + corr_feature[0],
                                "Drop " + corr_feature[1],
                                # "Combine with " + corr_feature[1],
                                "Do nothing",
                            ],
                            selected_action,
                        )

        if isinstance(self.target_type_, DiscreteTypedFeature):
            unique_vals = list(self.target_type_.meta_data["internal"]["counts"].keys())
            # binary classification dataset, suggest setting a positive class only if the values are not True/False
            if len(unique_vals) == 2 and True not in unique_vals:
                unique_vals.append("Do nothing")

                # for auto transform, do not suggest a default unless it is one of the known positive classes
                selected_action = unique_vals[2]

                # find positive label if recommendations can be manually updated
                if not self.auto_transform:
                    pos_vals = ["Y", "YES", "y", "Yes", "yes", "1", "true"]
                    pos_label_index = np.where(np.isin(pos_vals, unique_vals) == True)[
                        0
                    ]
                    selected_action = (
                        pos_vals[pos_label_index[0]]
                        if len(pos_label_index) > 0
                        else unique_vals[0]
                    )

                self._build_recommendation(
                    recommendations,
                    "positive_class",
                    [self.target_],
                    "Set Positive Class",
                    unique_vals,
                    selected_action,
                )
            # check if dataset is imbalanced
            if not self.is_balanced and self.fix_imbalance:
                target_value_counts = df[self.target_].value_counts()
                minority_class_len = min(
                    target_value_counts.iteritems(), key=lambda k: k[1]
                )[1]
                majority_class_len = max(
                    target_value_counts.iteritems(), key=lambda k: k[1]
                )[1]
                minor_majority_ratio = minority_class_len / majority_class_len

                # up-sample if length of dataframe is less than or equal to MAX_LEN_FOR_UP_SAMPLING = 5000
                # down-sample if minor_majority_ratio is greater than or equal to MIN_RATIO_FOR_DOWN_SAMPLING = 1/20
                if len(df) <= utils.MAX_LEN_FOR_UP_SAMPLING:
                    suggested_sampling = "Up-sample"
                elif minor_majority_ratio >= utils.MIN_RATIO_FOR_DOWN_SAMPLING:
                    suggested_sampling = "Down-sample"
                else:
                    suggested_sampling = "Do nothing"

                self._build_recommendation(
                    recommendations,
                    "fix_imbalance",
                    [self.target_],
                    "Imbalanced Target({0:.2%})".format(minor_majority_ratio),
                    ["Do nothing", "Down-sample", "Up-sample"],
                    suggested_sampling,
                )

        return recommendations

    def fit(self, X):
        self.reco_dict_ = self._get_recommendations(X)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        if self.reco_dict_ is None:
            self.fit(X)
        return self.transform(X, fit_transform=True, update_transformer_log=True)

    def transform(
        self,
        X,
        progress=DummyProgressBar(),
        fit_transform=False,
        update_transformer_log=False,
    ):
        df = self._transform(
            X,
            progress=progress,
            fit_transform=fit_transform,
            update_transformer_log=update_transformer_log,
        )

        # cleanup unused objects
        if hasattr(self, "reco_dict_") and not fit_transform:
            del self.reco_dict_
            del self.feature_metadata_
            del self.correlation_
            del self.feature_ranking_
        return df

    def transformer_log(self, action):
        """local wrapper to both log and record in the actions_performed array"""
        logger.info(action)
        self.actions_performed_.append(action)

    def _transform(
        self,
        X,
        progress=DummyProgressBar(),
        fit_transform=False,
        update_transformer_log=False,
    ):
        if hasattr(self, "reco_dict_") and len(self.reco_dict_) > 0:
            self.drop_columns = []
            # self.combine_columns = []
            # self.fill_nan_dict_ = {}
            columns_to_drop = []
            for recommendation_type_index in range(
                0, len(Recommendation.recommendation_types)
            ):
                recommendation_type = Recommendation.recommendation_types[
                    recommendation_type_index
                ]
                if recommendation_type in self.reco_dict_:
                    if recommendation_type_index == 0:
                        columns_to_drop = self.reco_dict_[recommendation_type]
                        self.drop_columns.extend(columns_to_drop)
                        if update_transformer_log:
                            self.transformer_log(
                                utils.wrap_lines(
                                    columns_to_drop, heading="Drop constant columns:"
                                )
                            )
                    elif recommendation_type == "positive_class":
                        value = self.reco_dict_[recommendation_type][self.target_][
                            "Selected Action"
                        ]
                        if value != "Do nothing":
                            if self.target_ in X.columns:
                                # X = X.set_positive_class(value, missing_value=False)
                                X[self.target_] = X[self.target_].map(
                                    lambda x: x == value
                                )
                                if update_transformer_log:
                                    self.transformer_log(
                                        "Set %s as positive class" % value
                                    )
                    elif recommendation_type == "fix_imbalance" and self.fix_imbalance:
                        self.balancing_strategy = self.reco_dict_[recommendation_type][
                            self.target_
                        ]["Selected Action"]
                        if update_transformer_log:
                            self.transformer_log(
                                "Fix imbalance using technique: %s"
                                % self.balancing_strategy
                            )
                    else:
                        # Get the new column name if it has been combined with another
                        for column in self.reco_dict_[recommendation_type]:
                            selected_action = self.reco_dict_[recommendation_type][
                                column
                            ]["Selected Action"]

                            if selected_action.startswith("Drop"):
                                if selected_action != "Drop":
                                    column = selected_action.split(" ", 1)[1]
                                if column not in columns_to_drop:
                                    self.drop_columns.append(column)
                                    if update_transformer_log:
                                        self.transformer_log(
                                            'Drop: "{}"'.format(column)
                                        )
                            # elif selected_action.startswith('Combine'):
                            #     column1 = column
                            #     column2 = selected_action.split(" ", 2)[2]

                            #     if column1 not in self.drop_columns and column2 not in self.drop_columns:
                            #         self.combine_columns.append((column1, column2))
                            #         if update_transformer_log:
                            #             self.transformer_log('Combine: "{}" with "{}"'.format(column1, column2))
                            elif recommendation_type == "imputation":
                                fill_val = (
                                    self.fill_nan_dict_[column]
                                    if selected_action
                                    == "Fill missing values with constant"
                                    else get_fill_val(
                                        self.feature_metadata_,
                                        column,
                                        selected_action,
                                        constant="constant",
                                    )
                                )
                                if fill_val is not None:
                                    if column == self.target_:
                                        # target fill need not be reproduced at the time of scoring,
                                        # as it won't be present
                                        self.fill_na_target_ = fill_val
                                    else:
                                        if update_transformer_log:
                                            self.transformer_log(
                                                '{} in {}: "{}"'.format(
                                                    selected_action, column, fill_val
                                                )
                                            )
                                        self.fill_nan_dict_[column] = fill_val

        # Drop columns
        if len(self.drop_columns) > 0:
            logger.info("Dropping columns " + str(set(self.drop_columns)))
        X = X.drop(self.drop_columns, axis=1)

        # fill na
        if len(self.fill_nan_dict_) > 0:
            logger.info(
                "Filling NaN values in " + str(list(self.fill_nan_dict_.keys()))
            )
        for col, fill_val in self.fill_nan_dict_.items():
            if col in X:
                if (
                    X[col].dtype.name == "category"
                    and fill_val not in X[col].cat.categories.tolist()
                ):
                    X[col] = X[col].cat.add_categories([fill_val])
                X[col] = X[col].fillna(fill_val)

        # fix imbalance only at the time of initial fit, the subsequent transform calls are used to
        # reproduce transformations, and sampling is not required at that time
        if fit_transform and self.fix_imbalance:
            if self.balancing_strategy and self.balancing_strategy != "Do nothing":
                progress.update("Fixing imbalance by %s" % self.balancing_strategy)
                # The imputation during transformation uses the sampled df to find the columns contains nan values.
                # This strategy could miss  some columns with very few nans in the larger df. The up_sample
                # takes care of filling out such missed nan values. Downsample is not affected
                X = (
                    up_sample(X, self.target_, feature_types=self.feature_metadata_)
                    if self.balancing_strategy == "Up-sample"
                    else down_sample(X, self.target_)
                )
            else:
                progress.update()
        else:
            progress.update()
        return X

    def _get_na_action(self, recommendations, column, null_counts, null_ratio):
        if isinstance(self.feature_metadata_[column], ContinuousTypedFeature):
            if null_ratio == 1:
                possible_actions = [
                    "Drop",
                    "Fill missing values with constant",
                    "Do nothing",
                ]
                selected_action = "Drop"
            else:
                possible_actions = [
                    "Drop",
                    "Fill missing values with mean",
                    "Fill missing values with median",
                    "Fill missing values with frequent",
                    "Fill missing values with constant",
                    "Do nothing",
                ]
                if null_ratio <= 0.4:
                    selected_action = "Fill missing values with mean"
                else:
                    selected_action = "Drop"
        else:
            if null_ratio == 1:
                possible_actions = [
                    "Drop",
                    "Fill missing values with constant",
                    "Do nothing",
                ]
                selected_action = "Drop"
            else:
                possible_actions = [
                    "Drop",
                    "Fill missing values with frequent",
                    "Fill missing values with constant",
                    "Do nothing",
                ]
                if null_ratio <= 0.4:
                    selected_action = "Fill missing values with frequent"
                else:
                    selected_action = "Drop"
        if null_ratio < 0.1:
            msg = "Contains missing values({0})".format(null_counts)
        else:
            msg = "Contains missing values({0:.2%})".format(null_ratio)
        self._build_recommendation(
            recommendations,
            "imputation",
            [column],
            msg,
            possible_actions,
            selected_action,
        )
