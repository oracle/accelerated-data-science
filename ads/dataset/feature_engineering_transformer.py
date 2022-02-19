#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import print_function, absolute_import

from sklearn.base import TransformerMixin

from ads.dataset.progress import DummyProgressBar


class FeatureEngineeringTransformer(TransformerMixin):
    def __init__(self, feature_metadata=None):
        self.feature_metadata_ = feature_metadata
        self.function_ = None
        self.function_kwargs_ = None

    def __repr__(self):
        return "No feature engineering transformations"

    def fit(self, X, y=None):
        self.function_ = None
        self.function_kwargs_ = None
        del self.feature_metadata_
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y=y).transform(X, fit_transform=True)

    def transform(self, df, progress=DummyProgressBar(), fit_transform=False):
        if self.function_ is not None:
            return df.pipe(self.function_, **self.function_kwargs_)
        return df
