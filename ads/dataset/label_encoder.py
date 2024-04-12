#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import bisect
import numpy as np

from collections import defaultdict
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder


class DataFrameLabelEncoder(TransformerMixin):
    """
    Label encoder for `pandas.DataFrame` and `dask.dataframe.core.DataFrame`.

    Attributes
    ----------
    label_encoders : defaultdict
        Holds the label encoder for each column.

    Examples
    --------
    >>> import pandas as pd
    >>> from ads.dataset.label_encoder import DataFrameLabelEncoder

    >>> df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    >>> le = DataFrameLabelEncoder()
    >>> le.fit_transform(X=df)

    """

    def __init__(self):
        """Initialize an instance of DataFrameLabelEncoder."""
        self.label_encoders = defaultdict(LabelEncoder)

    def fit(self, X: "pandas.DataFrame"):
        """
        Fits a DataFrameLabelEncoder.

        Parameters
        ----------
        X : pandas.DataFrame
            Target values.

        Returns
        -------
        self : returns an instance of self.
            Fitted label encoder.

        """
        for column in X.columns:
            if X[column].dtype.name in ["object", "category", "bool"]:
                X[column] = X[column].astype(str)
                self.label_encoders[column] = LabelEncoder()
                self.label_encoders[column].fit(X[column])
                label_encoder_classes_ = [
                    str(class_)
                    for class_ in self.label_encoders[column].classes_.tolist()
                ]
                bisect.insort_left(label_encoder_classes_, "unknown")
                label_encoder_classes_ = np.asarray(label_encoder_classes_)
                self.label_encoders[column].classes_ = label_encoder_classes_
        return self

    def transform(self, X: "pandas.DataFrame"):
        """
        Transforms a dataset using the DataFrameLabelEncoder.

        Parameters
        ----------
        X : pandas.DataFrame
            Target values.

        Returns
        -------
        pandas.DataFrame
            Labels as normalized encodings.

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
