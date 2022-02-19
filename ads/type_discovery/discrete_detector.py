#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import print_function, absolute_import, division

import pandas as pd
from sklearn.utils.multiclass import type_of_target

from ads.type_discovery import logger
from ads.type_discovery.abstract_detector import DiscreteDiscoveryDetector
from ads.type_discovery.typed_feature import (
    OrdinalTypedFeature,
    CategoricalTypedFeature,
)
from ads.common import utils


class DiscreteDetector(DiscreteDiscoveryDetector):

    _max_categorical_values = 100

    def _get_categorical_or_ordinal(self, name, series):
        #
        # categoricals are unordered discreet types
        # ordinals are ordered discreet (int) types
        #

        low_level_type_name = series.dtype.name

        if low_level_type_name == "category" or low_level_type_name == "bool":
            return "categorical"

        else:
            #
            # after removing nulls the new Series might already be categorical
            #
            nulls_removed = pd.Series(list(series.loc[~series.isna()]))
            if (
                nulls_removed.dtype.name == "category"
                or nulls_removed.dtype.name == "bool"
            ):
                return "categorical"

            count_distinct = series.nunique()
            observations = series.size

            tot = type_of_target(
                list(nulls_removed.head(min(nulls_removed.size, 2000)))
            )

            if tot == "binary":
                return "categorical"

            elif tot == "multiclass":
                if count_distinct <= DiscreteDetector._max_categorical_values:
                    if low_level_type_name in utils.numeric_pandas_dtypes():
                        return "ordinal"
                    else:
                        return "categorical"

                if low_level_type_name.startswith(
                    "int"
                ) or low_level_type_name.startswith("float"):
                    if nulls_removed.min() >= 0:
                        if (
                            low_level_type_name.startswith("int")
                            or nulls_removed.sum()
                            == nulls_removed.astype("int64").sum()
                        ):
                            return "ordinal"
                    # by summing all the values and summing all the int values we can know all the values are integers

        return False

    def discover(self, name, series):

        guessed_type = self._get_categorical_or_ordinal(
            name, series.loc[~series.isnull()]
        )

        if guessed_type == "categorical":
            logger.debug("column [{}]/[{}] categorical".format(name, series.dtype))
            return CategoricalTypedFeature.build(name, series)
        elif guessed_type == "ordinal":
            logger.debug("column [{}]/[{}] ordinal".format(name, series.dtype))
            return OrdinalTypedFeature.build(name, series)
        else:
            return False


if __name__ == "__main__":
    dd = DiscreteDetector()

    print(
        dd.discover(
            "str-categorical",
            pd.Series(["a", "a", "a", "b", "c", "a"], dtype="category"),
        )
    )
    print(
        dd.discover(
            "bool-categorical", pd.Series([True, False, True, True, True, None, True])
        )
    )
    print(dd.discover("continuous", pd.Series([None, 3.14, 12.0, 1, 2, 3, None])))
    print(
        dd.discover("int-1-categorical", pd.Series([1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 9]))
    )
    print(dd.discover("int-2-categorical", pd.Series([1, 1, 1, 5, 9])))
    print(dd.discover("real-3-categorical", pd.Series([1.0, 2.0, 3.0, 1.0, 4.0, 5.0])))
    print(
        dd.discover(
            "bool-categorical", pd.Series([True, False, True, True, True, None, True])
        )
    )
