#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import print_function, absolute_import

import numpy as np
import pandas as pd

from collections import defaultdict, Counter
from itertools import combinations, product


def _chi_squared(count_matrix: np.ndarray, n_obs: int) -> float:
    """
    Computes Chi-squared when given a contingency table
    """
    row_sums = np.tile(np.sum(count_matrix, axis=1), (count_matrix.shape[1], 1)).T
    col_sums = np.tile(np.sum(count_matrix, axis=0), (count_matrix.shape[0], 1))
    return np.sum(
        np.square(count_matrix - row_sums * col_sums / n_obs)
        / (row_sums * col_sums / n_obs)
    )


def _cramers_v(cat1: np.ndarray, cat2: np.ndarray) -> float:
    """
    Calculates the cramers v of two numpy arrays.
    """
    keep_cat1 = ~pd.isnull(cat1)
    cat1_no_nan = cat1[keep_cat1]
    cat2_no_nan = cat2[keep_cat1]
    keep_cat2 = ~pd.isnull(cat2_no_nan)
    cat1_no_nan = cat1_no_nan[keep_cat2]
    cat2_no_nan = cat2_no_nan[keep_cat2]
    n = len(cat1_no_nan)
    if n == 1:
        return 0
    contingency_table, r, k = _count_occurrence(cat1_no_nan, cat2_no_nan)

    if r == 0:
        return 0.0000

    chi2 = _chi_squared(contingency_table, n)
    phi2 = chi2 / n

    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - (np.square(r - 1)) / (n - 1)
    kcorr = k - (np.square(k - 1)) / (n - 1)
    denominator = min((kcorr - 1), (rcorr - 1))
    if denominator == 0:
        return np.nan
    return np.sqrt(phi2corr / denominator)


def _list_to_dataframe(
    name_list: list, corr_list: list, normal_form: bool
) -> pd.DataFrame:
    corr_dict = defaultdict(dict)
    for idx, corr in zip(name_list, corr_list):
        row_name = idx[0]
        col_name = idx[1]
        corr_dict[row_name][col_name] = corr_dict[col_name][row_name] = round(corr, 4)
        corr_dict[row_name][row_name] = corr_dict[col_name][col_name] = 1.0000
    correlation_matrix = pd.DataFrame.from_dict(corr_dict).sort_index()
    correlation_matrix = correlation_matrix.loc[:, correlation_matrix.index]
    if normal_form:
        data = []
        for (col1, col2), corr in correlation_matrix.stack().iteritems():
            data.append([col1, col2, round(corr, 4)])
        return pd.DataFrame(data, columns=["Column 1", "Column 2", "Value"])
    else:
        return correlation_matrix


def _count_occurrence(cat1: np.ndarray, cat2: np.ndarray) -> (np.ndarray, int, int):
    """
    Calculates the contingency table of two arrays.
    """
    occurrence_cnt = Counter([(x, y) for x, y in zip(cat1, cat2)])
    nunique_cat1 = np.unique(cat1[~pd.isnull(cat1)])
    nunique_cat2 = np.unique(cat2[~pd.isnull(cat2)])
    r = len(nunique_cat1)
    k = len(nunique_cat2)
    contingency_table = np.zeros((r, k))
    for row, num1 in enumerate(nunique_cat1):
        for col, num2 in enumerate(nunique_cat2):
            contingency_table[row, col] = occurrence_cnt[(num1, num2)]

    return contingency_table, r, k


def _correlation_ratio(cat: np.ndarray, cts: np.ndarray):
    """
    Calculates the correlation of a pair of a category feature and a continuous feature
    using correlation ratio when input are two numpy arrays.
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


def cat_vs_cont(
    df: pd.DataFrame, categorical_columns, continuous_columns, normal_form: bool = True
) -> pd.DataFrame:
    """
    Calculates the correlation of all pairs of categorical features and continuous features.
    """
    numerical_categorical_pairs = list(product(categorical_columns, continuous_columns))
    corr_list = []
    for col in numerical_categorical_pairs:
        corr_list.append(
            _correlation_ratio(np.array(df[col[0]].values), np.array(df[col[1]].values))
        )
    correlation_matrix = _list_to_dataframe(
        numerical_categorical_pairs, corr_list, normal_form=normal_form
    )
    return correlation_matrix


def cat_vs_cat(df: pd.DataFrame, normal_form: bool = True) -> pd.DataFrame:
    """
    Calculates the correlation of all pairs of categorical features and categorical features.
    """
    categorical_columns = df.columns.to_list()
    categorical_pairs = list(combinations(categorical_columns, 2))
    corr_list = []
    for col in categorical_pairs:
        corr_list.append(
            _cramers_v(np.array(df[col[0]].values), np.array(df[col[1]].values))
        )
    correlation_matrix = _list_to_dataframe(
        categorical_pairs, corr_list, normal_form=normal_form
    )
    return correlation_matrix


def cont_vs_cont(df: pd.DataFrame, normal_form: bool = True) -> pd.DataFrame:
    """
    Calculates the Pearson correlation between two columns of the DataFrame.
    """
    if not normal_form:
        return df.corr(method="pearson")
    data = []
    for (col1, col2), corr in df.corr(method="pearson").stack().iteritems():
        data.append([col1, col2, round(corr, 4)])
    return pd.DataFrame(data, columns=["Column 1", "Column 2", "Value"])
