#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import tempfile
from typing import Tuple

import pandas as pd
import pytest
from ads.dataset.dataset import ADSDataset
from ads.dataset.pipeline import TransformerPipeline
from ads.dataset.target import TargetVariable
import mock, sys


class TestADSDataset:
    @classmethod
    def setup_class(cls):
        cls.test_dir = tempfile.TemporaryDirectory()
        cls.storage_options = {"config": {"test": "value"}}

    @classmethod
    def teardown_class(cls):
        cls.test_dir.cleanup()

    @pytest.mark.parametrize(
        "test_file_name, expected_file_name",
        [
            ("test.h5", "test.h5"),
            ("*test.h5", "0test.h5"),
            ("test", "test.h5"),
            ("some/path/*/test.h5", "some/path/0/test.h5"),
        ],
    )
    def test_to_hdf(self, test_file_name, expected_file_name):
        """Tests saving data to Hierarchical Data Format (HDF) files."""
        test_file_path = os.path.join(self.test_dir.name, test_file_name)
        test_key = "df"

        test_df = pd.DataFrame([["1", "2"], ["3", "4"]], columns=["One", "Two"])
        ads_dataset = ADSDataset(test_df, test_df, None)
        new_file_path = ads_dataset.to_hdf(
            path=test_file_path, key=test_key, storage_options=self.storage_options
        )

        assert expected_file_name in new_file_path
        result_df = pd.read_hdf(new_file_path, test_key)
        assert result_df.equals(test_df)

    def test_TargetVariable_with_scipy_uninstalled(self):
        with mock.patch.dict(sys.modules, {"scipy": None}):
            with pytest.raises(ModuleNotFoundError):
                test_df = pd.DataFrame()
                TargetVariable(test_df, "", None)

    def test_initialize_dataset(self):
        employees = ADSDataset(
            df=pd.read_csv(self.get_data_path()),
            name="test_dataset",
            description="test_description",
            storage_options={'config':{},'region':'us-ashburn-1'}
        )
        assert isinstance(employees, ADSDataset)
        assert isinstance(employees.df, pd.DataFrame)
        assert isinstance(employees.shape, Tuple)
        assert employees.name == "test_dataset"
        assert employees.description == "test_description"
        assert "type_discovery" in employees.init_kwargs
        assert isinstance(employees.transformer_pipeline, TransformerPipeline)

    def test_from_dataframe(self):
        employees = ADSDataset.from_dataframe(
            df=pd.read_csv(self.get_data_path()),
            name="test_dataset",
            description="test_description",
            storage_options={'config':{},'region':'us-ashburn-1'}
        )
        assert isinstance(employees, ADSDataset)
        assert isinstance(employees.df, pd.DataFrame)
        assert isinstance(employees.shape, Tuple)
        assert employees.name == "test_dataset"
        assert employees.description == "test_description"
        assert "type_discovery" in employees.init_kwargs
        assert isinstance(employees.transformer_pipeline, TransformerPipeline)

    def get_data_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "data", "orcl_attrition.csv")
