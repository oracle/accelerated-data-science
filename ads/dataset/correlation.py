#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import print_function, absolute_import

import numpy as np
import pandas as pd

from ads.common import logger
from ads.dataset.exception import ValidationError
from collections import defaultdict, Counter
from itertools import product, combinations
from typing import Tuple


def _cat_vs_cat(df: pd.core.frame.DataFrame, categorical_columns: list) -> pd.DataFrame:
    """
    calc the correlation of all pairs of categorical features and categorical features
    """
    if not categorical_columns:
        return pd.DataFrame()
    categorical_pairs = list(combinations(categorical_columns, 2))
    corr_list = []
    for col in categorical_pairs:
        cat1_name = col[0]
        cat2_name = col[1]
        _check_if_same_type(df[col[0]], cat1_name)
        _check_if_same_type(df[col[1]], cat2_name)
        corr_list.append(
            _cramers_v(np.array(df[col[0]].values), np.array(df[col[1]].values))
        )
    correlation_matrix = _list_to_dataframe(categorical_pairs, corr_list)
    return correlation_matrix


def _cat_vs_cts(
    df: pd.core.frame.DataFrame, categorical_columns: list, continuous_columns: list
) -> pd.DataFrame:
    """
    calc the correlation of all pairs of categorical features and continuous features
    """
    numerical_categorical_pairs = list(product(categorical_columns, continuous_columns))
    corr_list = []
    for col in numerical_categorical_pairs:
        corr_list.append(
            _correlation_ratio(np.array(df[col[0]].values), np.array(df[col[1]].values))
        )
    correlation_matrix = _list_to_dataframe(numerical_categorical_pairs, corr_list)
    return correlation_matrix


def _list_to_dataframe(name_list: list, corr_list: list) -> pd.DataFrame:
    corr_dict = defaultdict(dict)
    for idx, corr in zip(name_list, corr_list):
        row_name = idx[0]
        col_name = idx[1]
        corr_dict[row_name][col_name] = corr_dict[col_name][row_name] = round(corr, 4)
        corr_dict[row_name][row_name] = corr_dict[col_name][col_name] = 1.0000
    correlation_matrix = pd.DataFrame.from_dict(corr_dict).sort_index()
    correlation_matrix = correlation_matrix.loc[:, correlation_matrix.index]
    return correlation_matrix


def _correlation_ratio(cat: np.ndarray, cts: np.ndarray):
    """
    calc the correlation of a pair of a categorical feature and a continuous feature
    using correlation ratio when input are two numpy arrays
    """
    keep_cts = ~pd.isnull(cts)
    cat_no_nan = cat[keep_cts]
    cts_no_nan = cts[keep_cts]

    keep_cat = ~pd.isnull(cat_no_nan)
    cat_no_none = cat_no_nan[keep_cat]
    cts_no_none = cts_no_nan[keep_cat]

    unq_cat, tags, group_count = np.unique(
        list(cat_no_none), return_inverse=1, return_counts=1
    )
    group_mean = np.bincount(tags, cts_no_none) / group_count
    overall_mean = np.nanmean(cts_no_none)
    n = len(cts_no_none)

    dispersion_within = np.dot(group_count, np.square(group_mean - overall_mean))
    dispersion_population = cts_no_none.var() * n
    ratio = dispersion_within / dispersion_population

    return np.sqrt(ratio)


def _count_occurrence(
    cat1: np.ndarray, cat2: np.ndarray
) -> Tuple[np.ndarray, int, int]:
    """
    calc the contingency table of two arrays
    """
    occurance_cnt = Counter([(x, y) for x, y in zip(cat1, cat2)])
    nunique_cat1 = np.unique(cat1[~pd.isnull(cat1)])
    nunique_cat2 = np.unique(cat2[~pd.isnull(cat2)])
    r = len(nunique_cat1)
    k = len(nunique_cat2)
    contigency_table = np.zeros((r, k))
    for row, num1 in enumerate(nunique_cat1):
        for col, num2 in enumerate(nunique_cat2):
            contigency_table[row, col] = occurance_cnt[(num1, num2)]

    return contigency_table, r, k


def _chi_squared(count_matrix: np.ndarray, n_obs: int) -> float:
    """
    Compute Chi-squared when given a contingency table
    """
    row_sums = np.tile(np.sum(count_matrix, axis=1), (count_matrix.shape[1], 1)).T
    col_sums = np.tile(np.sum(count_matrix, axis=0), (count_matrix.shape[0], 1))
    return np.sum(
        np.square(count_matrix - row_sums * col_sums / n_obs)
        / (row_sums * col_sums / n_obs)
    )


def _cramers_v(cat1: np.ndarray, cat2: np.ndarray) -> float:
    """
    calc the cramers v of two numpy arrays
    """
    n = len(cat1)
    if n == 1:
        return 0
    contigency_table, r, k = _count_occurrence(cat1, cat2)

    if r == 0:
        return 0.0000

    chi2 = _chi_squared(contigency_table, n)
    phi2 = chi2 / n

    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - (np.square(r - 1)) / (n - 1)
    kcorr = k - (np.square(k - 1)) / (n - 1)
    denominator = min((kcorr - 1), (rcorr - 1))
    if denominator == 0:
        return np.nan
    return np.sqrt(phi2corr / denominator)


def _get_columns_by_type(
    feature_types_df: pd.DataFrame, threshold: float = 0.8
) -> Tuple[list, list, list]:
    """
    return the categorical columns, continuous columns and columns of other types
    """
    missing = feature_types_df.loc[:, "missing_percentage"] > threshold
    not_missing = feature_types_df.loc[:, "missing_percentage"] <= threshold
    missing_columns = list(feature_types_df.loc[missing, "feature_name"].values)
    constant_columns = list(
        feature_types_df.loc[
            (feature_types_df.loc[:, "type"].isin(["constant"])), "feature_name"
        ].values
    )
    categorical_columns = list(
        feature_types_df.loc[
            (feature_types_df.loc[:, "type"].isin(["categorical", "zipcode"]))
            & not_missing,
            "feature_name",
        ].values
    )
    continuous_columns = list(
        feature_types_df.loc[
            (feature_types_df.loc[:, "type"].isin(["continuous", "ordinal"]))
            & not_missing,
            "feature_name",
        ].values
    )
    other_columns = list(
        set(feature_types_df.index.values)
        - set(categorical_columns)
        - set(continuous_columns)
        - set(missing_columns)
        - set(constant_columns)
    )

    if missing_columns:
        logger.info(
            f"The columns {missing_columns} are not included because more than {threshold}% of the values are missing. "
        )
        logger.info(
            f"The columns {missing_columns} are not included because more than {threshold}% of the values are missing. "
            f"Adjust this threshold using the `nan_threshold` parameter."
        )
    if constant_columns:
        logger.info(
            " The constant columns {} are not included.".format(constant_columns)
        )
    if other_columns:
        logger.info(
            f" The columns {other_columns} are not included because more than {threshold}% of the values are missing, or "
            f"they are not one of the following types: "
            f"`categorical`, `zipcode`, `continuous`"
            f", or `ordinal`."
        )

    return categorical_columns, continuous_columns, other_columns


def _validate_correlation_methods(correlation_methods):
    if isinstance(correlation_methods, str):
        correlation_methods = [correlation_methods]
    for method in correlation_methods:
        if method not in ["all", "pearson", "cramers v", "correlation ratio"]:
            raise ValidationError(f"{method} is not supported.")
    if "all" in [method for method in correlation_methods]:
        correlation_methods = ["pearson", "cramers v", "correlation ratio"]
    return correlation_methods


def _check_if_same_type(series, col_name):
    col = series.dropna().values
    if len(col) > 0:
        col_type = type(col[0])
        if not all([isinstance(x, col_type) for x in col]):
            raise TypeError(
                f"More than one data type in the column `{col_name}`. Keep all the values in that column the same type."
            )
