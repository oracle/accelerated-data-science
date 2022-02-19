#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict, List
from matplotlib.colors import LinearSegmentedColormap


def _sienna_light_to_dark_color_palette():
    return LinearSegmentedColormap.from_list("", ["#FDF3E4", "#512C1B"])


def _continuous_columns(feature_types: Dict) -> List[str]:
    """
    Parameters
    ----------
    feature_types : Dict
        Column name mapping to list of feature types ordered by most to least relevant.

    Returns
    -------
    List[str]
        List of columns that have continuous or ordinal in the feature type list.

    Note
    ____
    if a column has both ordinal/continuous and categorical pick whichever comes first.
    """
    continuous_cols = []
    for col in feature_types:
        for feature_type in feature_types[col]:
            if feature_type == "continuous" or feature_type == "ordinal":
                continuous_cols.append(col)
                break
            if feature_type == "category":
                break
    return continuous_cols


def _categorical_columns(feature_types: Dict) -> List[str]:
    """
    Parameters
    ----------
    feature_types : Dict
        column name mapping to list of feature types ordered by most to least relevant.

    Returns
    -------
    List[str]
        List of columns that have categorical in the feature type list.
    Note
    ____
    if a column has both ordinal/continuous and categorical pick whichever comes first.
    """
    categorical_cols = []
    for col in feature_types:
        for feature_type in feature_types[col]:
            if feature_type == "category":
                categorical_cols.append(col)
                break
            if feature_type == "continuous" or feature_type == "ordinal":
                break
    return categorical_cols
