#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import pandas as pd
import pytest
import unittest
from joblib import load
from ads.model import SklearnModel
from tests.integration.config import secrets


FILE_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model_deployment_test_files"
)
MODEL_PATH_ALL = os.path.join(FILE_FOLDER, "container_all/models/clf_lda.joblib")
MODEL_PATH_DEFAULT = os.path.join(
    FILE_FOLDER, "container_default/models/clf_lda.joblib"
)
ARTIFACT_DIR_ALL = os.path.join(FILE_FOLDER, "container_all/")
ARTIFACT_DIR_DEFAULT = os.path.join(FILE_FOLDER, "container_default/")
TEST_FILE = os.path.join(ARTIFACT_DIR_DEFAULT, "test.json")
COMPARTMENT_ID = secrets.common.COMPARTMENT_ID
PROJECT_ID = secrets.common.PROJECT_OCID
CONTAINER_IMAGE = secrets.model_deployment.MODEL_DEPLOYMENT_BYOC_IMAGE
CONTAINER_IMAGE_DEFAULT = secrets.model_deployment.MODEL_DEPLOYMENT_BYOC_IMAGE_DEFAULT
LOG_GROUP_ID = secrets.common.LOG_GROUP_ID
LOG_ID = secrets.model_deployment.LOG_OCID

test_cases = [
    {
        "wait_for_completion": True,
        "display_name": "generic_model_deployment_display_name",
        "description": "generic_model_deployment_description",
        "deployment_instance_shape": "VM.Standard.E4.Flex",
        "deployment_instance_count": 1,
        "deployment_bandwidth_mbps": 10,
        "deployment_log_group_id": LOG_GROUP_ID,
        "deployment_access_log_id": LOG_ID,
        "deployment_predict_log_id": LOG_ID,
        "deployment_memory_in_gbs": 16,
        "deployment_ocpus": 1,
        "deployment_image": CONTAINER_IMAGE,
        "compartment_id": COMPARTMENT_ID,
        "project_id": PROJECT_ID,
        "freeform_tags": {"test": "value"},
        "image_digest": "sha256:d6a63f8775c8bd5768d9fa0c7c3f51d2e03de49f5fa23f13f849b9c4d4c432f4",
        "entrypoint": ["python", "/opt/ds/model/deployed_model/api.py"],
        "server_port": 5000,
        "health_check_port": 5000,
        "environment_variables": {"test_key": "test_value"},
    },
    {
        "wait_for_completion": True,
        "deployment_image": CONTAINER_IMAGE_DEFAULT,
        "compartment_id": COMPARTMENT_ID,
        "project_id": PROJECT_ID,
    },
]


class GenericModelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        return super().tearDownClass()

    def test_generic_model_container(self):
        clf_lda = load(MODEL_PATH_ALL)
        sklearn_model = SklearnModel(clf_lda, artifact_dir=ARTIFACT_DIR_ALL)
        sklearn_model.prepare(
            model_file_name="clf_lda.joblib",
            ignore_conda_error=True,
            force_overwrite=True,
        )

        for key in [
            "ModelSerializationFormat",
            "ModelArtifacts",
            "SlugName",
            "CondaEnvironmentPath",
            "EnvironmentType",
        ]:
            with pytest.raises(ValueError, match=f"The metadata with {key} not found."):
                sklearn_model.metadata_custom.get(key)

        assert sklearn_model.metadata_taxonomy != None
        assert sklearn_model.metadata_provenance != None
        assert sklearn_model.metadata_provenance != None
        result = sklearn_model.verify(self.verify_data())
        assert result == {"prediction": [21]}

        model_from_artifact = SklearnModel.from_model_artifact(
            sklearn_model.artifact_dir,
            artifact_dir=sklearn_model.artifact_dir,
            force_overwrite=True,
            ignore_conda_error=True,
        )
        model_from_artifact_result = model_from_artifact.verify(self.verify_data())
        assert model_from_artifact_result == {"prediction": [21]}

        sklearn_model.save(compartment_id=COMPARTMENT_ID, project_id=PROJECT_ID)
        assert sklearn_model.model_id != None

        model_from_catalog = SklearnModel.from_model_catalog(
            sklearn_model.model_id,
            artifact_dir=sklearn_model.artifact_dir,
            force_overwrite=True,
            ignore_conda_error=True,
        )
        model_from_catalog_result = model_from_catalog.verify(self.verify_data())
        assert model_from_catalog_result == {"prediction": [21]}

        ## comment the deploy function to avoid reaching the limit count
        # sklearn_model.deploy(**test_cases[0])

        # assert sklearn_model.model_deployment.model_deployment_id != None
        # assert sklearn_model.model_deployment.url != None

        # result = sklearn_model.predict(data={"line": "12"}, auto_serialize_data=False)
        # assert result == {"prediction LDA": 21}

        # model_from_deploy = SklearnModel.from_model_deployment(
        #     sklearn_model.model_deployment_id,
        #     artifact_dir=sklearn_model.artifact_dir,
        #     force_overwrite=True,
        #     ignore_conda_error=True,
        # )
        # result_from_deployment = model_from_deploy.predict(
        #     data={"line": "12"}, auto_serialize_data=False
        # )
        # assert result_from_deployment == {"prediction LDA": 21}

        # sklearn_model.delete_deployment(wait_for_completion=True)
        # assert sklearn_model.model_deployment.lifecycle_state == "DELETED"

        sklearn_model.delete()

    def test_generic_model_container_default(self):
        clf_lda = load(MODEL_PATH_DEFAULT)
        sklearn_model = SklearnModel(clf_lda, artifact_dir=ARTIFACT_DIR_DEFAULT)
        sklearn_model.prepare(
            model_file_name="clf_lda.joblib",
            ignore_conda_error=True,
            force_overwrite=True,
        )

        for key in [
            "ModelSerializationFormat",
            "ModelArtifacts",
            "SlugName",
            "CondaEnvironmentPath",
            "EnvironmentType",
        ]:
            with pytest.raises(ValueError, match=f"The metadata with {key} not found."):
                sklearn_model.metadata_custom.get(key)

        assert sklearn_model.metadata_taxonomy != None
        assert sklearn_model.metadata_provenance != None
        assert sklearn_model.metadata_provenance != None

        result = sklearn_model.verify(self.verify_data())
        assert result == {"prediction": [21]}

        model_from_artifact = SklearnModel.from_model_artifact(
            sklearn_model.artifact_dir,
            artifact_dir=sklearn_model.artifact_dir,
            force_overwrite=True,
            ignore_conda_error=True,
        )
        model_from_artifact_result = model_from_artifact.verify(self.verify_data())
        assert model_from_artifact_result == {"prediction": [21]}

        sklearn_model.save(compartment_id=COMPARTMENT_ID, project_id=PROJECT_ID)
        assert sklearn_model.model_id != None

        model_from_catalog = SklearnModel.from_model_catalog(
            sklearn_model.model_id,
            artifact_dir=sklearn_model.artifact_dir,
            force_overwrite=True,
            ignore_conda_error=True,
        )
        model_from_catalog_result = model_from_catalog.verify(self.verify_data())
        assert model_from_catalog_result == {"prediction": [21]}

        ## comment the deploy function to avoid reaching the limit count
        # sklearn_model.deploy(**test_cases[1])

        # assert sklearn_model.model_deployment.model_deployment_id != None
        # assert sklearn_model.model_deployment.url != None

        # result = sklearn_model.predict(data={"line": "12"}, auto_serialize_data=False)
        # assert result == {"prediction LDA": 21}

        # model_from_deploy = SklearnModel.from_model_deployment(
        #     sklearn_model.model_deployment_id,
        #     artifact_dir=sklearn_model.artifact_dir,
        #     force_overwrite=True,
        #     ignore_conda_error=True,
        # )
        # result_from_deployment = model_from_deploy.predict(
        #     data={"line": "12"}, auto_serialize_data=False
        # )
        # assert result_from_deployment == {"prediction LDA": 21}

        # sklearn_model.delete_deployment(wait_for_completion=True)
        # assert sklearn_model.model_deployment.lifecycle_state == "DELETED"

        sklearn_model.delete()

    def test_generic_model_container_conda_env(self):
        clf_lda = load(MODEL_PATH_ALL)
        sklearn_model = SklearnModel(clf_lda, artifact_dir=ARTIFACT_DIR_ALL)
        sklearn_model.prepare(
            model_file_name="clf_lda.joblib",
            inference_conda_env="oci://service-conda-packs@id19sfcrra6z/service_pack/cpu/General_Machine_Learning_for_CPUs_on_Python_3.8/1.0/generalml_p38_cpu_v1",
            ignore_conda_error=True,
            force_overwrite=True,
        )

        for key in [
            "ModelSerializationFormat",
            "ModelArtifacts",
            "SlugName",
            "CondaEnvironmentPath",
            "EnvironmentType",
        ]:
            with pytest.raises(ValueError, match=f"The metadata with {key} not found."):
                sklearn_model.metadata_custom.get(key)

        assert sklearn_model.metadata_taxonomy != None
        assert sklearn_model.metadata_provenance != None
        assert sklearn_model.metadata_provenance != None
        assert sklearn_model.runtime_info != None

        result = sklearn_model.verify(self.verify_data())
        assert result == {"prediction": [21]}

        model_from_artifact = SklearnModel.from_model_artifact(
            sklearn_model.artifact_dir,
            artifact_dir=sklearn_model.artifact_dir,
            force_overwrite=True,
            ignore_conda_error=True,
        )
        model_from_artifact_result = model_from_artifact.verify(self.verify_data())
        assert model_from_artifact_result == {"prediction": [21]}

        sklearn_model.save(compartment_id=COMPARTMENT_ID, project_id=PROJECT_ID)
        assert sklearn_model.model_id != None

        model_from_catalog = SklearnModel.from_model_catalog(
            sklearn_model.model_id,
            artifact_dir=sklearn_model.artifact_dir,
            force_overwrite=True,
            ignore_conda_error=True,
        )
        model_from_catalog_result = model_from_catalog.verify(self.verify_data())
        assert model_from_catalog_result == {"prediction": [21]}

        ## comment the deploy function to avoid reaching the limit count
        # sklearn_model.deploy(**test_cases[0])

        # assert sklearn_model.model_deployment.model_deployment_id != None
        # assert sklearn_model.model_deployment.url != None

        # result = sklearn_model.predict(data={"line": "12"}, auto_serialize_data=False)
        # assert result == {"prediction LDA": 21}

        # model_from_deploy = SklearnModel.from_model_deployment(
        #     sklearn_model.model_deployment_id,
        #     artifact_dir=sklearn_model.artifact_dir,
        #     force_overwrite=True,
        #     ignore_conda_error=True,
        # )
        # result_from_deployment = model_from_deploy.predict(
        #     data={"line": "12"}, auto_serialize_data=False
        # )
        # assert result_from_deployment == {"prediction LDA": 21}

        # sklearn_model.delete_deployment(wait_for_completion=True)
        # assert sklearn_model.model_deployment.lifecycle_state == "DELETED"

        sklearn_model.delete()

    def test_generic_model_container_conda_env_default(self):
        clf_lda = load(MODEL_PATH_DEFAULT)
        sklearn_model = SklearnModel(clf_lda, artifact_dir=ARTIFACT_DIR_DEFAULT)
        sklearn_model.prepare(
            model_file_name="clf_lda.joblib",
            inference_conda_env="oci://service-conda-packs@id19sfcrra6z/service_pack/cpu/General_Machine_Learning_for_CPUs_on_Python_3.8/1.0/generalml_p38_cpu_v1",
            ignore_conda_error=True,
            force_overwrite=True,
        )

        for key in [
            "ModelSerializationFormat",
            "ModelArtifacts",
            "SlugName",
            "CondaEnvironmentPath",
            "EnvironmentType",
        ]:
            with pytest.raises(ValueError, match=f"The metadata with {key} not found."):
                sklearn_model.metadata_custom.get(key)

        assert sklearn_model.metadata_taxonomy != None
        assert sklearn_model.metadata_provenance != None
        assert sklearn_model.metadata_provenance != None
        assert sklearn_model.runtime_info != None

        result = sklearn_model.verify(self.verify_data())
        assert result == {"prediction": [21]}

        model_from_artifact = SklearnModel.from_model_artifact(
            sklearn_model.artifact_dir,
            artifact_dir=sklearn_model.artifact_dir,
            force_overwrite=True,
            ignore_conda_error=True,
        )
        model_from_artifact_result = model_from_artifact.verify(self.verify_data())
        assert model_from_artifact_result == {"prediction": [21]}

        sklearn_model.save(compartment_id=COMPARTMENT_ID, project_id=PROJECT_ID)
        assert sklearn_model.model_id != None

        model_from_catalog = SklearnModel.from_model_catalog(
            sklearn_model.model_id,
            artifact_dir=sklearn_model.artifact_dir,
            force_overwrite=True,
            ignore_conda_error=True,
        )
        model_from_catalog_result = model_from_catalog.verify(self.verify_data())
        assert model_from_catalog_result == {"prediction": [21]}

        ## comment the deploy function to avoid reaching the limit count
        # sklearn_model.deploy(**test_cases[1])

        # assert sklearn_model.model_deployment.model_deployment_id != None
        # assert sklearn_model.model_deployment.url != None

        # result = sklearn_model.predict(data={"line": "12"}, auto_serialize_data=False)
        # assert result == {"prediction LDA": 21}

        # model_from_deploy = SklearnModel.from_model_deployment(
        #     sklearn_model.model_deployment_id,
        #     artifact_dir=sklearn_model.artifact_dir,
        #     force_overwrite=True,
        #     ignore_conda_error=True,
        # )
        # result_from_deployment = model_from_deploy.predict(
        #     data={"line": "12"}, auto_serialize_data=False
        # )
        # assert result_from_deployment == {"prediction LDA": 21}

        # sklearn_model.delete_deployment(wait_for_completion=True)
        # assert sklearn_model.model_deployment.lifecycle_state == "DELETED"

        sklearn_model.delete()

    def verify_data(self):
        TEST_DATA = os.path.join(TEST_FILE)
        data = pd.read_json(TEST_DATA)
        data_test = data.transpose()
        X = data_test.drop(data_test.loc[:, "Line":"# Letter"].columns, axis=1)
        X_test = X.iloc[int("12"), :].values.reshape(1, -1)
        return X_test
