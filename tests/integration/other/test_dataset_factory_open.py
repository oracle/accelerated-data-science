#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common import auth
from ads.dataset.factory import DatasetFactory
from sklearn.datasets import make_classification
from tests.integration.config import secrets
import pandas as pd
import pytest


class TestDatasetFactoryOpen:
    """
    Contains integration test cases for ads.dataset.dataset
    """

    def setup_class(cls):
        X_big, y_big = make_classification(n_samples=600000, n_features=200)
        cls.df_big = pd.concat([pd.DataFrame(X_big), pd.DataFrame(y_big)], axis=0)

        X_small, y_small = make_classification(n_samples=600, n_features=200)
        cls.df_small = pd.concat([pd.DataFrame(X_small), pd.DataFrame(y_small)], axis=0)

        cls.storage_options = auth.default_signer()

    def test_small_data(self):
        ds = DatasetFactory.open(self.df_small)
        path = f"oci://{secrets.other.BUCKET_3}@{secrets.common.NAMESPACE}/test_small_data.csv"
        ds.to_csv(path, index=False, storage_options=self.storage_options)

        ds_from_cloud = DatasetFactory.open(path, storage_options=self.storage_options)
        assert ds_from_cloud.shape == ds.shape

    @pytest.mark.skip(reason="This test moves to ocifs")
    def test_large_data(self):
        ds = DatasetFactory.open(self.df_big)
        path = f"oci://{secrets.other.BUCKET_3}@{secrets.common.NAMESPACE}/test_large_data.csv"
        ds.to_csv(path, index=False, storage_options=self.storage_options)

        ds_from_cloud = DatasetFactory.open(
            path, storage_options=self.storage_options, error_bad_lines=False
        )
        assert ds_from_cloud.shape == ds.shape

    @pytest.mark.skip(
        reason="pd.read_excel(..sheet=None) reads all sheets and return a dict. currently open does not handle that."
    )
    def test_xlsx(self):
        path = f"oci://{secrets.other.BUCKET_3}@{secrets.common.NAMESPACE}/Spend_Summary.xlsx"
        ds_from_cloud = DatasetFactory.open(
            path, storage_options=self.storage_options, sheet_name=None
        )
        ## need to understand what the right output is supposed to be and add assertion
