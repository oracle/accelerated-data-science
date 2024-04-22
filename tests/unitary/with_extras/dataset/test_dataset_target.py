#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from mock import patch
from typing import Tuple
import os
import pandas as pd
import pytest

from ads.common import utils
from ads.dataset.classification_dataset import BinaryClassificationDataset
from ads.dataset.dataset_with_target import ADSDatasetWithTarget
from ads.dataset.pipeline import TransformerPipeline
from ads.dataset.target import TargetVariable


class TestADSDatasetTarget:
    def get_data_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "data", "orcl_attrition.csv")

    def test_initialize_dataset_target(self):
        employees = ADSDatasetWithTarget(
            df=pd.read_csv(self.get_data_path()),
            target="Attrition",
            name="test_dataset",
            description="test_description",
            storage_options={"config": {}, "region": "us-ashburn-1"},
        )

        assert isinstance(employees, ADSDatasetWithTarget)
        assert employees.name == "test_dataset"
        assert employees.description == "test_description"
        self.assert_dataset(employees)

    def test_dataset_target_from_dataframe(self):
        employees = ADSDatasetWithTarget.from_dataframe(
            df=pd.read_csv(self.get_data_path()),
            target="Attrition",
            storage_options={"config": {}, "region": "us-ashburn-1"},
        ).set_positive_class("Yes")

        assert isinstance(employees, BinaryClassificationDataset)
        self.assert_dataset(employees)

    def test_accessor_with_target(self):
        df=pd.read_csv(self.get_data_path())
        employees = df.ads.dataset_with_target(
            target="Attrition"
        )

        assert isinstance(employees, BinaryClassificationDataset)
        self.assert_dataset(employees)

    def test_accessor_with_target_error(self):
        df=pd.read_csv(self.get_data_path())
        wrong_column = "wrong_column"
        with pytest.raises(
            ValueError, match=f"{wrong_column} column doesn't exist in data frame. Specify a valid one instead."
        ):
            employees = df.ads.dataset_with_target(
                target=wrong_column
            )

    def assert_dataset(self, dataset):
        assert isinstance(dataset.df, pd.DataFrame)
        assert isinstance(dataset.shape, Tuple)
        assert isinstance(dataset.target, TargetVariable)
        assert dataset.target.type["type"] == "categorical"
        assert "type_discovery" in dataset.init_kwargs
        assert isinstance(dataset.transformer_pipeline, TransformerPipeline)

    def test_seggested_sampling_for_imbalanced_dataset(self):
        employees = ADSDatasetWithTarget.from_dataframe(
            df=pd.read_csv(self.get_data_path()),
            target="Attrition",
        ).set_positive_class("Yes")

        rt = employees._get_recommendations_transformer(
            fix_imbalance=True, correlation_threshold=1
        )
        rt.fit(employees)

        ## Assert with default setup for thresholds MAX_LEN_FOR_UP_SAMPLING and MIN_RATIO_FOR_DOWN_SAMPLING
        assert utils.MAX_LEN_FOR_UP_SAMPLING == 5000
        assert utils.MIN_RATIO_FOR_DOWN_SAMPLING == 1 / 20

        assert (
            rt.reco_dict_["fix_imbalance"]["Attrition"]["Message"]
            == "Imbalanced Target(33.33%)"
        )
        # up-sample if length of dataframe is less than or equal to MAX_LEN_FOR_UP_SAMPLING
        assert len(employees) < utils.MAX_LEN_FOR_UP_SAMPLING
        assert (
            rt.reco_dict_["fix_imbalance"]["Attrition"]["Selected Action"]
            == "Up-sample"
        )

        # manipulate MAX_LEN_FOR_UP_SAMPLING, MIN_RATIO_FOR_DOWN_SAMPLING to get other recommendations
        with patch("ads.common.utils.MAX_LEN_FOR_UP_SAMPLING", 5):
            assert utils.MAX_LEN_FOR_UP_SAMPLING == 5
            rt.fit(employees)
            # expect down-sample suggested, because minor_majority_ratio is greater than MIN_RATIO_FOR_DOWN_SAMPLING
            assert (
                rt.reco_dict_["fix_imbalance"]["Attrition"]["Selected Action"]
                == "Down-sample"
            )
            with patch("ads.common.utils.MIN_RATIO_FOR_DOWN_SAMPLING", 0.35):
                rt.fit(employees)
                # expect "Do nothing" with both MAX_LEN_FOR_UP_SAMPLING, MIN_RATIO_FOR_DOWN_SAMPLING tweaked for sampled_df
                assert (
                    rt.reco_dict_["fix_imbalance"]["Attrition"]["Selected Action"]
                    == "Do nothing"
                )
