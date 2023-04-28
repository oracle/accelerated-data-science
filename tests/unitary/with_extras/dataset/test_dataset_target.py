#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Tuple
import pandas as pd
from ads.dataset.classification_dataset import BinaryClassificationDataset
from ads.dataset.dataset_with_target import ADSDatasetWithTarget
from ads.dataset.pipeline import TransformerPipeline
from ads.dataset.target import TargetVariable


class TestADSDatasetTarget:
    def test_initialize_dataset_target(self):
        employees = ADSDatasetWithTarget(
            df=pd.read_csv("oci://hosted-ds-datasets@bigdatadatasciencelarge/synthetic/orcl_attrition.csv"),
            target="Attrition",
            name="test_dataset",
            description="test_description",
            storage_options={'config':{},'region':'us-ashburn-1'}
        )

        assert isinstance(employees, ADSDatasetWithTarget)
        assert isinstance(employees.df, pd.DataFrame)
        assert isinstance(employees.shape, Tuple)
        assert isinstance(employees.target, TargetVariable)
        assert employees.target.type["type"] == "categorical"
        assert employees.name == "test_dataset"
        assert employees.description == "test_description"
        assert "type_discovery" in employees.init_kwargs
        assert isinstance(employees.transformer_pipeline, TransformerPipeline)

    def test_dataset_target_from_dataframe(self):
        employees = ADSDatasetWithTarget.from_dataframe(
            df=pd.read_csv("oci://hosted-ds-datasets@bigdatadatasciencelarge/synthetic/orcl_attrition.csv"),
            target="Attrition",
            storage_options={'config':{},'region':'us-ashburn-1'}
        ).set_positive_class('Yes')

        assert isinstance(employees, BinaryClassificationDataset)
        assert isinstance(employees.df, pd.DataFrame)
        assert isinstance(employees.shape, Tuple)
        assert isinstance(employees.target, TargetVariable)
        assert employees.target.type["type"] == "categorical"
        assert "type_discovery" in employees.init_kwargs
        assert isinstance(employees.transformer_pipeline, TransformerPipeline)
