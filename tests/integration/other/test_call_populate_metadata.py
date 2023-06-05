#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import shutil
from unittest.mock import patch

import pytest
import yaml
from ads.catalog.model import ModelCatalog
from ads.common.model_export_util import prepare_generic_model
from tests.integration.config import secrets

tmp_model_dir = "/tmp/test_call_populate_metadata/model"


@pytest.fixture
def conda_file(tmpdir_factory):
    conda_file = tmpdir_factory.mktemp("conda")
    manifest = {
        "manifest": {
            "pack_path": "pack_path: oci://service-conda-packs@id19sfcrra6z/service_pack/cpu/pyspark/1.0/pyspv10",
            "python": "3.6",
            "slug": "pyspv10",
            "type": "data_science",
            "version": "1.0",
            "arch_type": "CPU",
            "manifest_version": "1.0",
            "name": "pyspark",
        }
    }
    with open(os.path.join(conda_file.strpath, "test_manifest.yaml"), "w") as mfile:
        yaml.dump(manifest, mfile)

    conda_prefix = os.environ["CONDA_PREFIX"]
    os.environ["CONDA_PREFIX"] = conda_file.strpath
    yield conda_file
    os.environ["CONDA_PREFIX"] = conda_prefix


class TestModelArtifactPopulateMetadata:
    """Contains test cases for ads.catalog.model.py"""

    @patch("ads.common.model_artifact._TRAINING_RESOURCE_OCID", None)
    def test_call_populate_metadata_when_save_model_artifact(self):
        if not os.path.exists(tmp_model_dir):
            os.makedirs(tmp_model_dir)
        generic_model_artifact = prepare_generic_model(
            tmp_model_dir,
            force_overwrite=True,
            ignore_deployment_error=True,
        )
        generic_model_artifact.populate_metadata()
        assert (
            "test.txt"
            not in generic_model_artifact.metadata_custom.get("ModelArtifacts").value
        )
        open(generic_model_artifact.artifact_dir + "/test.txt", "x")
        from ads.common import auth

        compartment_id = secrets.common.COMPARTMENT_ID
        project_id = secrets.common.PROJECT_OCID
        authorization = auth.default_signer()
        mc_model = generic_model_artifact.save(
            project_id=project_id,
            compartment_id=compartment_id,
            display_name="advanced-ds-test",
            auth=authorization,
            training_id=None,
        )
        assert (
            "test.txt"
            in generic_model_artifact.metadata_custom.get("ModelArtifacts").value
        )
        mc = ModelCatalog(compartment_id=compartment_id)
        assert mc.delete_model(mc_model) == True
        shutil.rmtree(tmp_model_dir)
