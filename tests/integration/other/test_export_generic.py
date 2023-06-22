#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import shutil
import tempfile

import oci
import pandas as pd
import pytest
from ads.common.model_export_util import prepare_generic_model
from ads.model.model_metadata import ModelCustomMetadata
from ads.common.object_storage_details import ObjectStorageDetails
from ads.feature_engineering.schema import Schema
from ads.common import auth
from ads.catalog.model import ModelCatalog
from sklearn import linear_model
from ads.model.model_metadata import Framework
from tests.integration.config import secrets


class TestPrepare:
    tmp_model_dir = "/tmp/model"

    train_X = [[1, 2], [2, 3], [3, 4], [4, 3]]
    train_y = [19, 26, 33, 30]
    gamma_reg_model = linear_model.GammaRegressor()
    AUTH = "api_key"
    BUCKET_NAME = secrets.other.BUCKET_3
    NAMESPACE = secrets.common.NAMESPACE
    OS_PREFIX = "unit_test"
    profile = "DEFAULT"
    oci_path = f"oci://{BUCKET_NAME}@{NAMESPACE}/{OS_PREFIX}"
    config_path = os.path.expanduser(os.path.join("~/.oci", "config"))

    compartment_id = secrets.common.COMPARTMENT_ID
    project_id = secrets.common.PROJECT_OCID
    authorization = auth.default_signer()

    if os.path.exists(config_path):
        config = oci.config.from_file(config_path, profile)
        # oci_client = ObjectStorageClient(config)
    else:
        raise Exception(f"OCI keys not found at {config_path}")
    storage_options = {"config": config}

    def setup_method(self):
        if not os.path.exists(self.tmp_model_dir):
            os.mkdir(self.tmp_model_dir)

    def test_prepare_artifact(self):
        prepare_generic_model(
            self.tmp_model_dir, force_overwrite=True, ignore_deployment_error=True
        )
        print(os.listdir(self.tmp_model_dir))
        assert len(os.listdir(self.tmp_model_dir)) > 0, "No files created"
        prepare_generic_model(
            self.tmp_model_dir,
            force_overwrite=True,
            fn_artifact_files_included=False,
            data_science_env=True,
            ignore_deployment_error=True,
        )
        assert "requirements.txt" not in os.listdir(self.tmp_model_dir)

    def test_metadata_integration(self):
        ma = prepare_generic_model(
            self.tmp_model_dir,
            X_sample=self.train_X,
            y_sample=self.train_y,
            model=self.gamma_reg_model,
            force_overwrite=True,
            ignore_deployment_error=True,
        )
        ma._save_data_from_memory(
            prefix=self.oci_path,
            train_data=pd.concat(
                [pd.DataFrame(self.train_X), pd.DataFrame(self.train_y)], axis=1
            ),
            train_data_name="training_data.csv",
            storage_options=self.storage_options,
        )
        print(os.listdir(self.tmp_model_dir))
        assert len(os.listdir(self.tmp_model_dir)) > 0, "No files created"
        assert isinstance(ma.metadata_custom, ModelCustomMetadata)
        assert "TrainingDataset" in ma.metadata_custom.keys
        assert (
            ma.metadata_custom["TrainingDataset"].value
            == f"oci://{secrets.other.BUCKET_3}@{secrets.common.NAMESPACE}/unit_test/training_data.csv"
        )
        assert ma.metadata_taxonomy.get("Framework").value == Framework.SCIKIT_LEARN

        ma.metadata_custom[
            "TrainingDataset"
        ].value = f"oci://{secrets.other.BUCKET_3}@{secrets.common.NAMESPACE}/unit_test/train.csv"
        assert (
            ma.metadata_custom["TrainingDataset"].value
            == f"oci://{secrets.other.BUCKET_3}@{secrets.common.NAMESPACE}/unit_test/train.csv"
        )

        assert isinstance(ma.schema_input, Schema)

        assert 0 in ma.schema_input.keys
        assert ma.schema_input[0].feature_type == "Integer"
        assert ma.schema_output[0].required
        ma.schema_input[0].required = False
        assert ma.schema_input[0].required == False
        assert "requirements.txt" not in os.listdir(self.tmp_model_dir)

        mc_model = ma.save(
            project_id=self.project_id,
            compartment_id=self.compartment_id,
            display_name="advanced-ds-test",
            description="A sample gamma regression classifier",
            ignore_pending_changes=True,
            auth=auth.default_signer(),
            training_id=None,
        )

        for key in mc_model.metadata_custom.keys:
            assert mc_model.metadata_custom.get(key) == ma.metadata_custom.get(key)
        for key in mc_model.metadata_taxonomy.keys:
            assert mc_model.metadata_taxonomy.get(key) == ma.metadata_taxonomy.get(key)
        assert mc_model.schema_input[0] == ma.schema_input[0]
        assert mc_model.schema_input[1] == ma.schema_input[1]
        assert mc_model.schema_output[0] == ma.schema_output[0]

        mc = ModelCatalog(
            compartment_id=self.compartment_id,
            ds_client_auth=self.authorization,
            identity_client_auth=self.authorization,
        )
        lr_model = mc.get_model(model_id=mc_model.id)

        for key in mc_model.metadata_custom.keys:
            assert mc_model.metadata_custom.get(key) == lr_model.metadata_custom.get(
                key
            )
        for key in mc_model.metadata_taxonomy.keys:
            assert mc_model.metadata_taxonomy.get(
                key
            ) == lr_model.metadata_taxonomy.get(key)
        assert mc_model.schema_input[0] == lr_model.schema_input[0]
        assert mc_model.schema_input[1] == lr_model.schema_input[1]
        assert mc_model.schema_output[0] == lr_model.schema_output[0]

        model_artifact = mc.download_model(
            target_dir=self.tmp_model_dir, model_id=mc_model.id, force_overwrite=True
        )
        for key in mc_model.metadata_custom.keys:
            assert mc_model.metadata_custom.get(
                key
            ) == model_artifact.metadata_custom.get(key)
        assert mc_model.schema_input[0] == model_artifact.schema_input[0]
        assert mc_model.schema_input[1] == model_artifact.schema_input[1]
        assert mc_model.schema_output[0] == model_artifact.schema_output[0]
        mc = ModelCatalog(compartment_id=self.compartment_id)
        assert mc.delete_model(mc_model) == True

    def test_generic_model_called_default_on_existing_folder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_fn_path = os.path.join(
                tmp_dir, "fn-model"
            )  # create path to fn-model that is already exists
            os.mkdir(tmp_fn_path)
            model_artifact = prepare_generic_model(
                self.tmp_model_dir, force_overwrite=True, ignore_deployment_error=True
            )
            r = repr(model_artifact)

        # file func.py doesn't exist because fn_artifact_files_included is set to False by default
        assert r.find("func.py") == -1
        assert r.find("func.yaml") == -1

    @pytest.mark.skip(
        reason="Support of Fn is limited, code not removed yet, but customers not encouraged to use fn"
        "deployment. fdk removed from install_required in setup.py. This test not required. "
        "Remove when fn code be removed."
    )
    def test_generic_model_called_on_existing_folder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_fn_path = os.path.join(
                tmp_dir, "fn-model"
            )  # create path to fn-model that is already exists
            os.mkdir(tmp_fn_path)
            model_artifact = prepare_generic_model(
                self.tmp_model_dir,
                fn_artifact_files_included=True,
                force_overwrite=True,
                ignore_deployment_error=True,
            )
            r = repr(model_artifact)
        assert r.find("func.py") > 0  # file func.py exists in fn-model folder
        assert r.find("func.yaml") > 0  # file func.yaml exists in fn-model folder

    def test_generic_model_called_on_existing_folder_with_deprecated_arg1(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_fn_path = os.path.join(
                tmp_dir, "fn-model"
            )  # create path to fn-model that is already exists
            os.mkdir(tmp_fn_path)
            with pytest.raises(ValueError):
                model_artifact = prepare_generic_model(
                    self.tmp_model_dir,
                    function_artifacts=True,
                    force_overwrite=True,
                    ignore_deployment_error=True,
                )

    def test_generic_model_called_on_existing_folder_with_deprecated_arg2(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_fn_path = os.path.join(
                tmp_dir, "fn-model"
            )  # create path to fn-model that is already exists
            os.mkdir(tmp_fn_path)
            with pytest.raises(ValueError):
                model_artifact = prepare_generic_model(
                    self.tmp_model_dir,
                    function_artifacts=False,
                    fn_artifact_files_included=True,
                    force_overwrite=True,
                    ignore_deployment_error=True,
                )

    def test_generic_model_called_on_existing_folder_with_deprecated_arg3(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_fn_path = os.path.join(
                tmp_dir, "fn-model"
            )  # create path to fn-model that is already exists
            os.mkdir(tmp_fn_path)
            with pytest.raises(ValueError):
                model_artifact = prepare_generic_model(
                    self.tmp_model_dir,
                    function_artifacts=True,
                    fn_artifact_files_included=True,
                    force_overwrite=True,
                    ignore_deployment_error=True,
                )

    def test_generic_model_with_python_version_specified(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_fn_path = os.path.join(
                tmp_dir, "fn-model"
            )  # create path to fn-model that is already exists
            os.mkdir(tmp_fn_path)
            model_artifact = prepare_generic_model(
                self.tmp_model_dir,
                inference_python_version="3.7",
                force_overwrite=True,
                ignore_deployment_error=True,
                inference_conda_env="oci://bucketname",
            )
            r = repr(model_artifact)

            with open(os.path.join(model_artifact.artifact_dir, "runtime.yaml")) as f:
                assert "INFERENCE_PYTHON_VERSION" in f.read()
            assert r.find("runtime.yaml") > 1

    @pytest.mark.skip(
        reason="Support of Fn is limited, code not removed yet, but customers not encouraged to use fn"
        "deployment. fdk removed from install_required in setup.py. This test not required. "
        "Remove when fn code be removed."
    )
    def test_generic_model_called_on_existing_folder_with_force_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_fn_path = os.path.join(
                tmp_dir, "fn-model"
            )  # create path to fn-model that is already exists
            os.mkdir(tmp_fn_path)
            # create fn artifacts in the model path
            model_artifact = prepare_generic_model(
                self.tmp_model_dir,
                fn_artifact_files_included=True,
                force_overwrite=True,
                ignore_deployment_error=True,
            )

            # do not expect fn artifacts in the model path with fn_artifact_files_included = False
            model_artifact2 = prepare_generic_model(
                self.tmp_model_dir, force_overwrite=True, ignore_deployment_error=True
            )
            r = repr(model_artifact2)

        assert r.find("func.py") >= 0
        assert r.find("func.yaml") >= 0

    def test_oci_url_get_parts(self):
        inference_conda_env = (
            "oci://service-conda-packs@ociodscdev/service_pack/cpu/Data Exploration and "
            "Manipulation for CPU Python 3.7/1.0/dataexpl_p37_cpu_v1"
        )
        bucket_name, namespace, object_name = ObjectStorageDetails.from_path(
            inference_conda_env
        ).to_tuple()
        assert bucket_name == "service-conda-packs"
        assert namespace == "ociodscdev"
        assert (
            object_name == "service_pack/cpu/Data Exploration and Manipulation for "
            "CPU Python 3.7/1.0/dataexpl_p37_cpu_v1"
        )

    def teardown_class(cls):
        shutil.rmtree(cls.tmp_model_dir)
