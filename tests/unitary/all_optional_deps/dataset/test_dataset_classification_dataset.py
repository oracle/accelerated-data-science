#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd
import sklearn.datasets

from ads.dataset.classification_dataset import BinaryClassificationDataset


class TestBinaryClassificationDataset:
    X, y = sklearn.datasets.make_classification(n_samples=1000, n_features=10)
    df = pd.DataFrame(data=X, columns=[str(i) for i in range(X.shape[1])])
    df["target"] = y

    def test_binary_classification_dataset_init_with_positive_class_set(self):
        """Validate mapping for calculation of positive class."""
        assert set(self.df["target"].unique()) == set([0, 1])
        dataset = BinaryClassificationDataset(
            df=self.df,
            sampled_df=self.df,
            target="target",
            target_type={"target": "category"},
            shape=self.X.shape,
            positive_class=1,
        )
        assert set(dataset["target"].unique()) == set([False, True])
