#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np
from sklearn import preprocessing

from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.common import utils
from ads.type_discovery.typed_feature import CategoricalTypedFeature
from ads.type_discovery.typed_feature import ContinuousTypedFeature
from ads.type_discovery.typed_feature import OrdinalTypedFeature

ABSOLUTE_SKEWNESS_THRESHOLD = 0.5
# between 0 and 1, ratio of least / most represented class
CLASS_IMBALANCE_THRESHOLD = 0.5


class TargetVariable:
    """
    This class provides target specific APIs.
    """

    @runtime_dependency(module="scipy", install_from=OptionalDependency.VIZ)
    def __init__(self, sampled_ds, target, target_type):
        from scipy.stats import skew

        self.sampled_ds = sampled_ds
        self.name = target
        self.type = target_type
        self.target_vals = None
        if isinstance(self.type, CategoricalTypedFeature):
            self.target_vals = self.sampled_ds.sampled_df[target].unique().tolist()
            val_counts = self.sampled_ds.sampled_df[target].value_counts()
            self.class_imbalance_ratio = val_counts.min() / val_counts.max()
        elif isinstance(self.type, ContinuousTypedFeature):
            try:
                self.skewness = np.abs(self.sampled_ds.sampled_df[target].skew())
            except TypeError as e:
                self.skewness = np.abs(
                    self.sampled_ds.sampled_df[target].astype("float").skew()
                )
        else:
            # can also be DateTimeTypedFeature IPAddressTypedFeature PhoneNumberTypedFeature GISTypedFeature
            # AddressTypedFeature DocumentTypedFeature ZipcodeTypedFeature UnknownTypedFeature ConstantTypedFeature
            # CreditCardTypedFeature OrdinalTypedFeature
            self.skewness = None
        self.numeric_columns = (
            self.sampled_ds.sampled_df._get_numeric_data().columns.values
        )

    def show_in_notebook(self, feature_names=None):  # pragma: no cover
        """
        Plot target distribution or target versus feature relation.

        Parameters
        ----------
        feature_names: list, Optional
            Plot target against a list of features.
            Display target distribution if feature_names is not provided.
        """
        if not utils.is_notebook():
            print("show_in_notebook called but not in notebook environment")
            return

        verbose = True
        if feature_names is not None:
            for feature_name in feature_names:
                self.sampled_ds.plot(
                    feature_name, self.name, verbose=verbose
                ).show_in_notebook()
                verbose = False
        else:
            self.sampled_ds.plot(self.name, verbose=False).show_in_notebook()

    def is_balanced(
        self,
        skewness_threshold=ABSOLUTE_SKEWNESS_THRESHOLD,
        class_imbalance_threshold=CLASS_IMBALANCE_THRESHOLD,
    ):
        """
        Returns True if the target is balanced, False otherwise.

        Returns
        -------
        is_balanced: bool
        """
        if isinstance(self.type, ContinuousTypedFeature):
            return self.skewness < skewness_threshold
        elif isinstance(self.type, CategoricalTypedFeature):
            return self.class_imbalance_ratio > class_imbalance_threshold
        else:
            return True
