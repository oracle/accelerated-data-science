#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
Contains tests for ads.common.model_artifact.save_from_memory and
"""
import os
from unittest.mock import patch

import numpy as np
import oci
import pandas as pd
import pytest

from ads.common.model_artifact import ModelArtifact
from ads.common import auth
import tempfile
from oci.object_storage import ObjectStorageClient
import shutil
from tests.integration.config import secrets


class TestArtifactSaveData:
    array = np.array([[1, 2], [2, 3], [3, 4], [4, 3]])
    df = pd.DataFrame([[1, 2], [2, 3], [3, 4], [4, 3]])
    l = [[1, 2], [2, 3], [3, 4], [4, 3]]
    file_name = "data.csv"

    def setup_class(cls):
        cls.model_dir = tempfile.mkdtemp(prefix="model")
        cls.model_artifact = ModelArtifact(
            cls.model_dir, reload=False, create=True, ignore_deployment_error=True
        )
        cls.df.to_csv(os.path.join(cls.model_dir, cls.file_name))
        cls.compartment_id = secrets.common.COMPARTMENT_ID
        cls.project_id = secrets.common.PROJECT_OCID
        cls.authorization = auth.default_signer()

        cls.BUCKET_NAME = "ads_test"
        cls.NAMESPACE = secrets.common.NAMESPACE
        cls.OS_PREFIX = "unit_test"
        cls.oci_path = f"oci://{cls.BUCKET_NAME}@{cls.NAMESPACE}/{cls.OS_PREFIX}"

        cls.oci_client = ObjectStorageClient(**cls.authorization)

        datafiles = cls.oci_client.list_objects(
            namespace_name=cls.NAMESPACE,
            bucket_name=cls.BUCKET_NAME,
            prefix=cls.OS_PREFIX,
            fields="name,etag,md5",
        )
        for item in datafiles.data.objects:
            try:
                cls.oci_client.delete_object(
                    namespace_name=cls.NAMESPACE,
                    bucket_name=cls.BUCKET_NAME,
                    object_name=item.name,
                )
            except:
                pass

    @pytest.mark.parametrize("data", [array, df, l])
    def test_modelartifact_save_data_from_memory(self, data):
        self.model_artifact._save_data_from_memory(
            prefix=f"oci://{self.BUCKET_NAME}@{self.NAMESPACE}/{self.OS_PREFIX}",
            train_data=data,
            train_data_name=f"{type(data)}_train.csv",
            validation_data=data,
            validation_data_name=f"{type(data)}_validation.csv",
            storage_options=self.authorization,
        )
        datafiles = self.oci_client.list_objects(
            namespace_name=self.NAMESPACE,
            bucket_name=self.BUCKET_NAME,
            prefix=self.OS_PREFIX,
            fields="name,etag,md5",
        )
        assert os.path.join(self.OS_PREFIX, f"{type(data)}_train.csv") in [
            item.name for item in datafiles.data.objects
        ]
        assert os.path.join(self.OS_PREFIX, f"{type(data)}_validation.csv") in [
            item.name for item in datafiles.data.objects
        ]
        assert set(
            [
                "TrainingDatasetSize",
                "ValidationDatasetSize",
                "TrainingDataset",
                "ValidationDataset",
            ]
        ) == set(self.model_artifact.metadata_custom.keys)
        assert self.model_artifact.metadata_custom.get(
            "TrainingDataset"
        ).value == os.path.join(self.oci_path, f"{type(data)}_train.csv")
        assert self.model_artifact.metadata_custom.get(
            "ValidationDataset"
        ).value == os.path.join(self.oci_path, f"{type(data)}_validation.csv")
        assert (
            self.model_artifact.metadata_custom.get("TrainingDatasetSize").value
            == "(4, 2)"
        )

    def test_modelartifact_save_data_from_files(self):
        self.model_artifact._save_data_from_file(
            f"oci://{self.BUCKET_NAME}@{self.NAMESPACE}/{self.OS_PREFIX}",
            train_data_path=os.path.join(self.model_dir, self.file_name),
            storage_options=self.authorization,
        )
        datafiles = self.oci_client.list_objects(
            namespace_name=self.NAMESPACE,
            bucket_name=self.BUCKET_NAME,
            prefix=self.OS_PREFIX,
            fields="name,etag,md5",
        )
        assert os.path.join(self.OS_PREFIX, self.file_name) in [
            item.name for item in datafiles.data.objects
        ]
        assert self.model_artifact.metadata_custom.get(
            "TrainingDataset"
        ).value == os.path.join(self.oci_path, "data.csv")

    def test_modelartifact_save_from_files_oci_path(self):
        self.model_artifact._save_data_from_file(self.oci_path, data_type="validation")
        assert (
            self.model_artifact.metadata_custom.get("ValidationDataset").value
            == self.oci_path
        )

    @patch("ads.common.model_artifact._TRAINING_RESOURCE_OCID", None)
    def test_modelartifact_save_with_training_id(self):
        mc_model = self.model_artifact.save(
            project_id=self.project_id,
            compartment_id=self.compartment_id,
            display_name="advanced-ds-test",
            description="None",
            ignore_pending_changes=True,
            auth=self.authorization,
            training_id=None,
        )
        assert mc_model.provenance_metadata.training_id == None

    def test_modelartifact_save_with_tags(self):
        mc_model = self.model_artifact.save(
            project_id=self.project_id,
            compartment_id=self.compartment_id,
            display_name="advanced-ds-test",
            description="None",
            ignore_pending_changes=True,
            auth=self.authorization,
            training_id=None,
            freeform_tags={"freeform_key": "freeform_val"},
            defined_tags={"teamcity-test": {"CreatedBy": "test_user"}},
        )
        assert mc_model.freeform_tags == {"freeform_key": "freeform_val"}
        assert mc_model.defined_tags["teamcity-test"]["CreatedBy"] == "test_user"

    def teardown_class(cls):
        if os.path.exists(cls.model_dir):
            shutil.rmtree(cls.model_dir, ignore_errors=True)
