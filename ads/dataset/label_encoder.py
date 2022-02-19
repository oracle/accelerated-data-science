#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import bisect
from collections import defaultdict

from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder


class DataFrameLabelEncoder(TransformerMixin):
    """
    Label encoder for pandas.dataframe. dask.dataframe.core.DataFrame
    """

    def __init__(self):
        self.label_encoders = defaultdict(LabelEncoder)

    def fit(self, X):
        """
        Fits a DataFrameLAbelEncoder.
        """
        for column in X.columns:
            if X[column].dtype.name in ["object", "category"]:
                X[column] = X[column].astype(str)
                self.label_encoders[column] = LabelEncoder()
                self.label_encoders[column].fit(X[column])
                label_encoder_classes_ = [
                    str(class_)
                    for class_ in self.label_encoders[column].classes_.tolist()
                ]
                bisect.insort_left(label_encoder_classes_, "unknown")
                self.label_encoders[column].classes_ = label_encoder_classes_
        return self

    def transform(self, X):
        """
        Transforms a dataset using the DataFrameLAbelEncoder.
        """
        categorical_columns = list(self.label_encoders.keys())
        if len(categorical_columns) == 0:
            return X

        def _label_encode_with_unknown(name, series):
            return self.label_encoders[name].transform(
                series.map(lambda x: str(x)).map(
                    lambda x: x
                    if x in self.label_encoders[name].classes_
                    else "unknown"
                )
            )

        X[categorical_columns] = X[categorical_columns].apply(
            lambda series: _label_encode_with_unknown(series.name, series)
        )
        return X
