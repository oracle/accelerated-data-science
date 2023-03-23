#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module with all default warnings provided to user. These are registered to relevant feature
types directly in the feature type files themselves.
"""

import pandas as pd


def missing_values_handler(s: pd.Series) -> pd.DataFrame:
    """
    Warning for > 5 percent missing values (Nans) in series.

    Parameters
    ----------
    s : pd.Series
        Pandas series - column of some feature type.

    Returns
    -------
    pd.Dataframe
        Dataframe with 4 columns 'Warning', 'Message', 'Metric', 'Value'
        and 2 rows, where first row is count of missing values and second is
        percentage of missing values.
    """
    num_missing = s.isna().sum()
    pct_missing = 100 * num_missing / len(s)
    df = pd.DataFrame([], columns=["Warning", "Message", "Metric", "Value"])
    if pct_missing >= 5:
        df.loc[0] = ["missing", f"{num_missing} missing values", "count", num_missing]
        df.loc[1] = [
            "missing",
            f"{pct_missing:.1f}% missing values",
            "percentage",
            round(pct_missing, 2),
        ]
    return df


def skew_handler(s: pd.Series) -> pd.DataFrame:
    """
    Warning if absolute value of skew is greater than 1.

    Parameters
    ----------
    s : pd.Series
        Pandas series - column of some feature type, expects continuous values.

    Returns
    -------
    pd.Dataframe
        Dataframe with 4 columns 'Warning', 'Message', 'Metric', 'Value'
        and 1 rows, which lists skew value of that column.
    """
    series_skew = s.skew()
    df = pd.DataFrame([], columns=["Warning", "Message", "Metric", "Value"])
    if abs(series_skew) > 1:
        df.loc[0] = ["skew", f"{series_skew:.3f} skew", "skew", round(series_skew, 2)]
    return df


def high_cardinality_handler(s: pd.Series) -> pd.DataFrame:
    """
    Warning if number of unique values (including Nan) in series is greater than or equal to 15.

    Parameters
    ----------
    s : pd.Series
        Pandas series - column of some feature type.

    Returns
    -------
    pd.Dataframe
        Dataframe with 4 columns 'Warning', 'Message', 'Metric', 'Value'
        and 1 rows, which lists count of unique values.
    """
    num_unique = s.nunique(dropna=False)
    df = pd.DataFrame([], columns=["Warning", "Message", "Metric", "Value"])
    if num_unique == len(s):
        df.loc[0] = [
            "high-cardinality",
            f"every value is distinct",
            "count",
            num_unique,
        ]
    elif num_unique >= 15:
        df.loc[0] = [
            "high-cardinality",
            f"{num_unique} unique values",
            "count",
            num_unique,
        ]
    return df


def zeros_handler(s: pd.Series) -> pd.DataFrame:
    """
    Warning for greater than 10 percent zeros in series.

    Parameters
    ----------
    s : pd.Series
        Pandas series - column of some feature type.

    Returns
    -------
    pd.Dataframe
        Dataframe with 4 columns 'Warning', 'Message', 'Metric', 'Value'
        and 2 rows, where first row is count of zero values and second is
        percentage of zero values.
    """
    num_zeros = (s == 0).sum()
    pct_missing = 100 * num_zeros / len(s)
    df = pd.DataFrame([], columns=["Warning", "Message", "Metric", "Value"])
    if pct_missing >= 5:
        df.loc[0] = ["zeros", f"{num_zeros} zeros", "count", num_zeros]
        df.loc[1] = [
            "zeros",
            f"{pct_missing:.1f}% zeros",
            "percentage",
            round(pct_missing, 2),
        ]
    return df
