#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
Contains tests for ads.common.model_artifact.save
"""
import os
import shutil
import tempfile
from unittest.mock import patch

import numpy as np
import oci
import pandas as pd
import pytest
from ads.common import auth
from ads.common.model_artifact import ModelArtifact
from ads.model.model_introspect import IntrospectionNotPassed, ModelIntrospect
from oci.object_storage import ObjectStorageClient
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

    def setup_method(self):
        self.AUTH = "api_key"
        self.BUCKET_NAME = "ads_test"
        self.NAMESPACE = secrets.common.NAMESPACE
        self.OS_PREFIX = "unit_test"
        profile = "DEFAULT"
        self.oci_path = f"oci://{self.BUCKET_NAME}@{self.NAMESPACE}/{self.OS_PREFIX}"

        config_path = os.path.expanduser(os.path.join("~/.oci", "config"))
        if os.path.exists(config_path):
            self.config = oci.config.from_file(config_path, profile)
            self.oci_client = ObjectStorageClient(self.config)
        else:
            raise Exception(f"OCI keys not found at {config_path}")

    @patch.object(ModelIntrospect, "_reset")
    def test_modelartifact_save_with_introspection(self, mock_reset):
        with pytest.raises(IntrospectionNotPassed) as execinfo:
            mc_model = self.model_artifact.save(
                project_id=self.project_id,
                compartment_id=self.compartment_id,
                display_name="advanced-ds-test",
                description="None",
                ignore_pending_changes=True,
                auth=self.authorization,
                training_id=None,
                ignore_introspection=False,
            )

    def teardown_method(self):
        datafiles = self.oci_client.list_objects(
            namespace_name=self.NAMESPACE,
            bucket_name=self.BUCKET_NAME,
            prefix=self.OS_PREFIX,
            fields="name,etag,md5",
        )
        for item in datafiles.data.objects:
            try:
                self.oci_client.delete_object(
                    namespace_name=self.NAMESPACE,
                    bucket_name=self.BUCKET_NAME,
                    object_name=item.name,
                )
            except:
                pass

    def teardown_class(cls):
        if os.path.exists(cls.model_dir):
            shutil.rmtree(cls.model_dir, ignore_errors=True)
