#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import json
import os
import unittest
from dataclasses import asdict
from importlib import reload
from unittest.mock import MagicMock, patch

import oci
import pytest
from parameterized import parameterized

import ads.aqua.modeldeployment.deployment
import ads.config
from ads.aqua.modeldeployment import AquaDeploymentApp, MDInferenceResponse
from ads.aqua.modeldeployment.entities import (
    AquaDeployment,
    AquaDeploymentDetail,
    ModelParams,
)
from ads.aqua.common.errors import AquaRuntimeError, AquaValueError
from ads.model.datascience_model import DataScienceModel
from ads.model.deployment.model_deployment import ModelDeployment
from ads.model.model_metadata import ModelCustomMetadata

null = None


class TestDataset:
    SERVICE_COMPARTMENT_ID = "ocid1.compartment.oc1..<OCID>"
    USER_COMPARTMENT_ID = "ocid1.compartment.oc1..<USER_COMPARTMENT_OCID>"
    COMPARTMENT_ID = "ocid1.compartment.oc1..<UNIQUE_OCID>"
    MODEL_DEPLOYMENT_ID = "ocid1.datasciencemodeldeployment.oc1.<region>.<MD_OCID>"
    MODEL_DEPLOYMENT_URL = "https://modeldeployment.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.<region>.<MD_OCID>"
    MODEL_ID = "ocid1.datasciencemodeldeployment.oc1.<region>.<MODEL_OCID>"
    DEPLOYMENT_IMAGE_NAME = "dsmc://image-name:1.0.0.0"
    DEPLOYMENT_SHAPE_NAME = "VM.GPU.A10.1"
    DEPLOYMENT_SHAPE_NAME_CPU = "VM.Standard.A1.Flex"

    model_deployment_object = [
        {
            "category_log_details": oci.data_science.models.CategoryLogDetails(
                **{
                    "access": oci.data_science.models.LogDetails(
                        **{
                            "log_group_id": "ocid1.loggroup.oc1.<region>.<OCID>",
                            "log_id": "ocid1.log.oc1.<region>.<OCID>",
                        }
                    ),
                    "predict": oci.data_science.models.LogDetails(
                        **{
                            "log_group_id": "ocid1.loggroup.oc1.<region>.<OCID>",
                            "log_id": "ocid1.log.oc1.<region>.<OCID>",
                        }
                    ),
                }
            ),
            "compartment_id": "ocid1.compartment.oc1..<OCID>",
            "created_by": "ocid1.user.oc1..<OCID>",
            "defined_tags": {},
            "description": "Mock description",
            "display_name": "model-deployment-name",
            "freeform_tags": {"OCI_AQUA": "active", "aqua_model_name": "model-name"},
            "id": "ocid1.datasciencemodeldeployment.oc1.<region>.<MD_OCID>",
            "lifecycle_state": "ACTIVE",
            "model_deployment_configuration_details": oci.data_science.models.SingleModelDeploymentConfigurationDetails(
                **{
                    "deployment_type": "SINGLE_MODEL",
                    "environment_configuration_details": oci.data_science.models.OcirModelDeploymentEnvironmentConfigurationDetails(
                        **{
                            "cmd": [],
                            "entrypoint": [],
                            "environment_configuration_type": "OCIR_CONTAINER",
                            "environment_variables": {
                                "BASE_MODEL": "service_models/model-name/artifact",
                                "MODEL_DEPLOY_ENABLE_STREAMING": "true",
                                "MODEL_DEPLOY_PREDICT_ENDPOINT": "/v1/completions",
                                "PARAMS": "--served-model-name odsc-llm --seed 42",
                            },
                            "health_check_port": 8080,
                            "image": "dsmc://image-name:1.0.0.0",
                            "image_digest": "sha256:mock22373c16f2015f6f33c5c8553923cf8520217da0bd9504471c5e53cbc9d",
                            "server_port": 8080,
                        }
                    ),
                    "model_configuration_details": oci.data_science.models.ModelConfigurationDetails(
                        **{
                            "bandwidth_mbps": 10,
                            "instance_configuration": oci.data_science.models.InstanceConfiguration(
                                **{
                                    "instance_shape_name": DEPLOYMENT_SHAPE_NAME,
                                    "model_deployment_instance_shape_config_details": null,
                                }
                            ),
                            "model_id": "ocid1.datasciencemodel.oc1.<region>.<OCID>",
                            "scaling_policy": oci.data_science.models.FixedSizeScalingPolicy(
                                **{"instance_count": 1, "policy_type": "FIXED_SIZE"}
                            ),
                        }
                    ),
                }
            ),
            "model_deployment_url": MODEL_DEPLOYMENT_URL,
            "project_id": "ocid1.datascienceproject.oc1.<region>.<OCID>",
            "time_created": "2024-01-01T00:00:00.000000+00:00",
        }
    ]

    model_deployment_object_gguf = [
        {
            "compartment_id": "ocid1.compartment.oc1..<OCID>",
            "created_by": "ocid1.user.oc1..<OCID>",
            "defined_tags": {},
            "description": "Mock description",
            "display_name": "model-deployment-name",
            "freeform_tags": {"OCI_AQUA": "active", "aqua_model_name": "model-name"},
            "id": "ocid1.datasciencemodeldeployment.oc1.<region>.<MD_OCID>",
            "lifecycle_state": "ACTIVE",
            "model_deployment_configuration_details": oci.data_science.models.SingleModelDeploymentConfigurationDetails(
                **{
                    "deployment_type": "SINGLE_MODEL",
                    "environment_configuration_details": oci.data_science.models.OcirModelDeploymentEnvironmentConfigurationDetails(
                        **{
                            "cmd": [],
                            "entrypoint": [],
                            "environment_configuration_type": "OCIR_CONTAINER",
                            "environment_variables": {
                                "BASE_MODEL": "service_models/model-name/artifact",
                                "BASE_MODEL_FILE": "model-name.gguf",
                                "MODEL_DEPLOY_ENABLE_STREAMING": "true",
                                "MODEL_DEPLOY_PREDICT_ENDPOINT": "/v1/completions",
                                "MODEL_DEPLOY_HEALTH_ENDPOINT": "/v1/models",
                            },
                            "health_check_port": 8080,
                            "image": "dsmc://image-name:1.0.0.0",
                            "image_digest": "sha256:mock22373c16f2015f6f33c5c8553923cf8520217da0bd9504471c5e53cbc9d",
                            "server_port": 8080,
                        }
                    ),
                    "model_configuration_details": oci.data_science.models.ModelConfigurationDetails(
                        **{
                            "bandwidth_mbps": 10,
                            "instance_configuration": oci.data_science.models.InstanceConfiguration(
                                **{
                                    "instance_shape_name": DEPLOYMENT_SHAPE_NAME_CPU,
                                    "model_deployment_instance_shape_config_details": oci.data_science.models.ModelDeploymentInstanceShapeConfigDetails(
                                        **{
                                            "ocpus": 10,
                                            "memory_in_gbs": 60.0,
                                        }
                                    ),
                                }
                            ),
                            "model_id": "ocid1.datasciencemodel.oc1.<region>.<OCID>",
                            "scaling_policy": oci.data_science.models.FixedSizeScalingPolicy(
                                **{"instance_count": 1, "policy_type": "FIXED_SIZE"}
                            ),
                        }
                    ),
                }
            ),
            "model_deployment_url": MODEL_DEPLOYMENT_URL,
            "project_id": "ocid1.datascienceproject.oc1.<region>.<OCID>",
            "time_created": "2024-01-01T00:00:00.000000+00:00",
        }
    ]

    model_deployment_object_tei_byoc = [
        {
            "category_log_details": oci.data_science.models.CategoryLogDetails(
                **{
                    "access": oci.data_science.models.LogDetails(
                        **{
                            "log_group_id": "ocid1.loggroup.oc1.<region>.<OCID>",
                            "log_id": "ocid1.log.oc1.<region>.<OCID>",
                        }
                    ),
                    "predict": oci.data_science.models.LogDetails(
                        **{
                            "log_group_id": "ocid1.loggroup.oc1.<region>.<OCID>",
                            "log_id": "ocid1.log.oc1.<region>.<OCID>",
                        }
                    ),
                }
            ),
            "compartment_id": "ocid1.compartment.oc1..<OCID>",
            "created_by": "ocid1.user.oc1..<OCID>",
            "defined_tags": {},
            "description": "Mock description",
            "display_name": "model-deployment-name",
            "freeform_tags": {"OCI_AQUA": "active", "aqua_model_name": "model-name"},
            "id": "ocid1.datasciencemodeldeployment.oc1.<region>.<MD_OCID>",
            "lifecycle_state": "ACTIVE",
            "model_deployment_configuration_details": oci.data_science.models.SingleModelDeploymentConfigurationDetails(
                **{
                    "deployment_type": "SINGLE_MODEL",
                    "environment_configuration_details": oci.data_science.models.OcirModelDeploymentEnvironmentConfigurationDetails(
                        **{
                            "cmd": [
                                "--model-id",
                                "/opt/ds/model/deployed_model/service_models/model-name/artifact/",
                                "--port",
                                "8080",
                            ],
                            "entrypoint": [],
                            "environment_configuration_type": "OCIR_CONTAINER",
                            "environment_variables": {
                                "BASE_MODEL": "service_models/model-name/artifact",
                                "MODEL_DEPLOY_PREDICT_ENDPOINT": "/v1/embeddings",
                            },
                            "health_check_port": 8080,
                            "image": "dsmc://image-name:1.0.0.0",
                            "image_digest": "sha256:mock22373c16f2015f6f33c5c8553923cf8520217da0bd9504471c5e53cbc9d",
                            "server_port": 8080,
                        }
                    ),
                    "model_configuration_details": oci.data_science.models.ModelConfigurationDetails(
                        **{
                            "bandwidth_mbps": 10,
                            "instance_configuration": oci.data_science.models.InstanceConfiguration(
                                **{
                                    "instance_shape_name": DEPLOYMENT_SHAPE_NAME,
                                    "model_deployment_instance_shape_config_details": null,
                                }
                            ),
                            "model_id": "ocid1.datasciencemodel.oc1.<region>.<OCID>",
                            "scaling_policy": oci.data_science.models.FixedSizeScalingPolicy(
                                **{"instance_count": 1, "policy_type": "FIXED_SIZE"}
                            ),
                        }
                    ),
                }
            ),
            "model_deployment_url": MODEL_DEPLOYMENT_URL,
            "project_id": "ocid1.datascienceproject.oc1.<region>.<OCID>",
            "time_created": "2024-01-01T00:00:00.000000+00:00",
        }
    ]

    aqua_deployment_object = {
        "id": "ocid1.datasciencemodeldeployment.oc1.<region>.<MD_OCID>",
        "display_name": "model-deployment-name",
        "aqua_service_model": False,
        "aqua_model_name": "model-name",
        "state": "ACTIVE",
        "description": "Mock description",
        "created_on": "2024-01-01T00:00:00.000000+00:00",
        "created_by": "ocid1.user.oc1..<OCID>",
        "endpoint": MODEL_DEPLOYMENT_URL,
        "private_endpoint_id": "",
        "environment_variables": {
            "BASE_MODEL": "service_models/model-name/artifact",
            "MODEL_DEPLOY_ENABLE_STREAMING": "true",
            "MODEL_DEPLOY_PREDICT_ENDPOINT": "/v1/completions",
            "PARAMS": "--served-model-name odsc-llm --seed 42",
        },
        "cmd": [],
        "console_link": "https://cloud.oracle.com/data-science/model-deployments/ocid1.datasciencemodeldeployment.oc1.<region>.<MD_OCID>?region=region-name",
        "lifecycle_details": "",
        "shape_info": {
            "instance_shape": DEPLOYMENT_SHAPE_NAME,
            "instance_count": 1,
            "ocpus": null,
            "memory_in_gbs": null,
        },
        "tags": {"OCI_AQUA": "active", "aqua_model_name": "model-name"},
    }

    aqua_deployment_gguf_env_vars = {
        "BASE_MODEL": "service_models/model-name/artifact",
        "BASE_MODEL_FILE": "model-name.gguf",
        "MODEL_DEPLOY_ENABLE_STREAMING": "true",
        "MODEL_DEPLOY_HEALTH_ENDPOINT": "/v1/models",
        "MODEL_DEPLOY_PREDICT_ENDPOINT": "/v1/completions",
    }

    aqua_deployment_gguf_shape_info = {
        "instance_shape": DEPLOYMENT_SHAPE_NAME_CPU,
        "instance_count": 1,
        "ocpus": 10,
        "memory_in_gbs": 60.0,
    }

    aqua_deployment_detail = {
        **vars(AquaDeployment(**aqua_deployment_object)),
        "log_group": {
            "id": "ocid1.loggroup.oc1.<region>.<OCID>",
            "name": "log-group-name",
            "url": "https://cloud.oracle.com/logging/log-groups/ocid1.loggroup.oc1.<region>.<OCID>?region=region-name",
        },
        "log": {
            "id": "ocid1.log.oc1.<region>.<OCID>",
            "name": "log-name",
            "url": "https://cloud.oracle.com/logging/search?searchQuery=search \"ocid1.compartment.oc1..<OCID>/ocid1.loggroup.oc1.<region>.<OCID>/ocid1.log.oc1.<region>.<OCID>\" | source='ocid1.datasciencemodeldeployment.oc1.<region>.<MD_OCID>' | sort by datetime desc&regions=region-name",
        },
    }

    model_params = {
        "model": "odsc-llm",
        "max_tokens": 500,
        "temperature": 0.8,
        "top_p": 0.8,
        "top_k": 10,
    }

    aqua_deployment_tei_byoc_embeddings_env_vars = {
        "BASE_MODEL": "service_models/model-name/artifact",
        "MODEL_DEPLOY_PREDICT_ENDPOINT": "/v1/embeddings",
    }

    aqua_deployment_tei_byoc_embeddings_shape_info = {
        "instance_shape": DEPLOYMENT_SHAPE_NAME,
        "instance_count": 1,
        "ocpus": None,
        "memory_in_gbs": None,
    }

    aqua_deployment_tei_byoc_embeddings_cmd = [
        "--model-id",
        "/opt/ds/model/deployed_model/service_models/model-name/artifact/",
        "--port",
        "8080",
    ]


class TestAquaDeployment(unittest.TestCase):
    def setUp(self):
        self.app = AquaDeploymentApp()
        self.app.region = "region-name"

    @classmethod
    def setUpClass(cls):
        cls.curr_dir = os.path.dirname(os.path.abspath(__file__))
        os.environ["CONDA_BUCKET_NS"] = "test-namespace"
        os.environ["ODSC_MODEL_COMPARTMENT_OCID"] = TestDataset.SERVICE_COMPARTMENT_ID
        os.environ["PROJECT_COMPARTMENT_OCID"] = TestDataset.USER_COMPARTMENT_ID
        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.modeldeployment.deployment)

    @classmethod
    def tearDownClass(cls):
        cls.curr_dir = None
        os.environ.pop("CONDA_BUCKET_NS", None)
        os.environ.pop("ODSC_MODEL_COMPARTMENT_OCID", None)
        os.environ.pop("PROJECT_COMPARTMENT_OCID", None)
        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.modeldeployment.deployment)

    def test_list_deployments(self):
        """Tests the list method in the AquaDeploymentApp class."""

        self.app.list_resource = MagicMock(
            return_value=[
                oci.data_science.models.ModelDeploymentSummary(**item)
                for item in TestDataset.model_deployment_object
            ]
        )
        results = self.app.list()
        received_args = self.app.list_resource.call_args.kwargs

        assert received_args.get("compartment_id") == TestDataset.USER_COMPARTMENT_ID
        assert len(results) == 1
        expected_attributes = AquaDeployment.__annotations__.keys()
        for r in results:
            actual_attributes = asdict(r)
            assert set(actual_attributes) == set(
                expected_attributes
            ), "Attributes mismatch"

    @patch("ads.aqua.modeldeployment.deployment.get_resource_name")
    def test_get_deployment(self, mock_get_resource_name):
        """Tests the get method in the AquaDeploymentApp class."""

        model_deployment = copy.deepcopy(TestDataset.model_deployment_object[0])
        self.app.ds_client.get_model_deployment = MagicMock(
            return_value=oci.response.Response(
                status=200,
                request=MagicMock(),
                headers=MagicMock(),
                data=oci.data_science.models.ModelDeploymentSummary(**model_deployment),
            )
        )
        mock_get_resource_name.side_effect = lambda param: (
            "log-group-name"
            if param.startswith("ocid1.loggroup")
            else "log-name"
            if param.startswith("ocid1.log")
            else ""
        )

        result = self.app.get(model_deployment_id=TestDataset.MODEL_DEPLOYMENT_ID)

        expected_attributes = set(AquaDeploymentDetail.__annotations__.keys()) | set(
            AquaDeployment.__annotations__.keys()
        )
        actual_attributes = asdict(result)
        assert set(actual_attributes) == set(expected_attributes), "Attributes mismatch"
        assert actual_attributes == TestDataset.aqua_deployment_detail
        assert result.log.name == "log-name"
        assert result.log_group.name == "log-group-name"

    def test_get_deployment_missing_tags(self):
        """Test for returning a runtime error if OCI_AQUA tag is missing."""
        with pytest.raises(
            AquaRuntimeError,
            match=f"Target deployment {TestDataset.MODEL_DEPLOYMENT_ID} is not Aqua deployment.",
        ):
            model_deployment = copy.deepcopy(TestDataset.model_deployment_object[0])
            model_deployment["freeform_tags"] = {}
            self.app.ds_client.get_model_deployment = MagicMock(
                return_value=oci.response.Response(
                    status=200,
                    request=MagicMock(),
                    headers=MagicMock(),
                    data=oci.data_science.models.ModelDeploymentSummary(
                        **model_deployment
                    ),
                )
            )

            self.app.get(model_deployment_id=TestDataset.MODEL_DEPLOYMENT_ID)

    def test_get_deployment_config(self):
        """Test for fetching config details for a given deployment."""

        config_json = os.path.join(
            self.curr_dir, "test_data/deployment/deployment_config.json"
        )
        with open(config_json, "r") as _file:
            config = json.load(_file)

        self.app.get_config = MagicMock(return_value=config)
        result = self.app.get_deployment_config(TestDataset.MODEL_ID)
        assert result == config

        self.app.get_config = MagicMock(return_value=None)
        result = self.app.get_deployment_config(TestDataset.MODEL_ID)
        assert result == None

    @patch("ads.aqua.modeldeployment.deployment.get_container_config")
    @patch("ads.aqua.model.AquaModelApp.create")
    @patch("ads.aqua.modeldeployment.deployment.get_container_image")
    @patch("ads.model.deployment.model_deployment.ModelDeployment.deploy")
    def test_create_deployment_for_foundation_model(
        self,
        mock_deploy,
        mock_get_container_image,
        mock_create,
        mock_get_container_config,
    ):
        """Test to create a deployment for foundational model"""
        aqua_model = os.path.join(
            self.curr_dir, "test_data/deployment/aqua_foundation_model.yaml"
        )
        mock_create.return_value = DataScienceModel.from_yaml(uri=aqua_model)
        config_json = os.path.join(
            self.curr_dir, "test_data/deployment/deployment_config.json"
        )
        with open(config_json, "r") as _file:
            config = json.load(_file)

        self.app.get_deployment_config = MagicMock(return_value=config)

        freeform_tags = {"ftag1": "fvalue1", "ftag2": "fvalue2"}
        defined_tags = {"dtag1": "dvalue1", "dtag2": "dvalue2"}

        container_index_json = os.path.join(
            self.curr_dir, "test_data/ui/container_index.json"
        )
        with open(container_index_json, "r") as _file:
            container_index_config = json.load(_file)
        mock_get_container_config.return_value = container_index_config

        mock_get_container_image.return_value = TestDataset.DEPLOYMENT_IMAGE_NAME
        aqua_deployment = os.path.join(
            self.curr_dir, "test_data/deployment/aqua_create_deployment.yaml"
        )
        model_deployment_obj = ModelDeployment.from_yaml(uri=aqua_deployment)
        model_deployment_dsc_obj = copy.deepcopy(TestDataset.model_deployment_object[0])
        model_deployment_dsc_obj["lifecycle_state"] = "CREATING"
        model_deployment_dsc_obj["defined_tags"] = defined_tags
        model_deployment_dsc_obj["freeform_tags"].update(freeform_tags)
        model_deployment_obj.dsc_model_deployment = (
            oci.data_science.models.ModelDeploymentSummary(**model_deployment_dsc_obj)
        )
        mock_deploy.return_value = model_deployment_obj

        result = self.app.create(
            model_id=TestDataset.MODEL_ID,
            instance_shape=TestDataset.DEPLOYMENT_SHAPE_NAME,
            display_name="model-deployment-name",
            log_group_id="ocid1.loggroup.oc1.<region>.<OCID>",
            access_log_id="ocid1.log.oc1.<region>.<OCID>",
            predict_log_id="ocid1.log.oc1.<region>.<OCID>",
            freeform_tags=freeform_tags,
            defined_tags=defined_tags,
        )

        mock_create.assert_called_with(
            model_id=TestDataset.MODEL_ID,
            compartment_id=None,
            project_id=None,
            freeform_tags=freeform_tags,
            defined_tags=defined_tags,
        )
        mock_get_container_image.assert_called()
        mock_deploy.assert_called()

        expected_attributes = set(AquaDeployment.__annotations__.keys())
        actual_attributes = asdict(result)
        assert set(actual_attributes) == set(expected_attributes), "Attributes mismatch"
        expected_result = copy.deepcopy(TestDataset.aqua_deployment_object)
        expected_result["state"] = "CREATING"
        expected_result["tags"].update(freeform_tags)
        expected_result["tags"].update(defined_tags)
        assert actual_attributes == expected_result

    @patch("ads.aqua.modeldeployment.deployment.get_container_config")
    @patch("ads.aqua.model.AquaModelApp.create")
    @patch("ads.aqua.modeldeployment.deployment.get_container_image")
    @patch("ads.model.deployment.model_deployment.ModelDeployment.deploy")
    def test_create_deployment_for_fine_tuned_model(
        self,
        mock_deploy,
        mock_get_container_image,
        mock_create,
        mock_get_container_config,
    ):
        """Test to create a deployment for fine-tuned model"""

        aqua_model = os.path.join(
            self.curr_dir, "test_data/deployment/aqua_finetuned_model.yaml"
        )
        datascience_model = DataScienceModel.from_yaml(uri=aqua_model)
        mock_create.return_value = datascience_model

        config_json = os.path.join(
            self.curr_dir, "test_data/deployment/deployment_config.json"
        )
        with open(config_json, "r") as _file:
            config = json.load(_file)

        self.app.get_deployment_config = MagicMock(return_value=config)

        container_index_json = os.path.join(
            self.curr_dir, "test_data/ui/container_index.json"
        )
        with open(container_index_json, "r") as _file:
            container_index_config = json.load(_file)
        mock_get_container_config.return_value = container_index_config

        mock_get_container_image.return_value = TestDataset.DEPLOYMENT_IMAGE_NAME
        aqua_deployment = os.path.join(
            self.curr_dir, "test_data/deployment/aqua_create_deployment.yaml"
        )
        model_deployment_obj = ModelDeployment.from_yaml(uri=aqua_deployment)
        model_deployment_dsc_obj = copy.deepcopy(TestDataset.model_deployment_object[0])
        model_deployment_dsc_obj["lifecycle_state"] = "CREATING"
        model_deployment_obj.dsc_model_deployment = (
            oci.data_science.models.ModelDeploymentSummary(**model_deployment_dsc_obj)
        )
        mock_deploy.return_value = model_deployment_obj

        result = self.app.create(
            model_id=TestDataset.MODEL_ID,
            instance_shape=TestDataset.DEPLOYMENT_SHAPE_NAME,
            display_name="model-deployment-name",
            log_group_id="ocid1.loggroup.oc1.<region>.<OCID>",
            access_log_id="ocid1.log.oc1.<region>.<OCID>",
            predict_log_id="ocid1.log.oc1.<region>.<OCID>",
        )

        mock_create.assert_called_with(
            model_id=TestDataset.MODEL_ID,
            compartment_id=None,
            project_id=None,
            freeform_tags=None,
            defined_tags=None,
        )
        mock_get_container_image.assert_called()
        mock_deploy.assert_called()

        expected_attributes = set(AquaDeployment.__annotations__.keys())
        actual_attributes = asdict(result)
        assert set(actual_attributes) == set(expected_attributes), "Attributes mismatch"
        expected_result = copy.deepcopy(TestDataset.aqua_deployment_object)
        expected_result["state"] = "CREATING"
        assert actual_attributes == expected_result

    @patch("ads.aqua.modeldeployment.deployment.get_container_config")
    @patch("ads.aqua.model.AquaModelApp.create")
    @patch("ads.aqua.modeldeployment.deployment.get_container_image")
    @patch("ads.model.deployment.model_deployment.ModelDeployment.deploy")
    def test_create_deployment_for_gguf_model(
        self,
        mock_deploy,
        mock_get_container_image,
        mock_create,
        mock_get_container_config,
    ):
        """Test to create a deployment for fine-tuned model"""

        aqua_model = os.path.join(
            self.curr_dir, "test_data/deployment/aqua_foundation_model.yaml"
        )
        datascience_model = DataScienceModel.from_yaml(uri=aqua_model)
        mock_create.return_value = datascience_model

        config_json = os.path.join(
            self.curr_dir, "test_data/deployment/deployment_config.json"
        )
        with open(config_json, "r") as _file:
            config = json.load(_file)

        self.app.get_deployment_config = MagicMock(return_value=config)

        container_index_json = os.path.join(
            self.curr_dir, "test_data/ui/container_index.json"
        )
        with open(container_index_json, "r") as _file:
            container_index_config = json.load(_file)
        mock_get_container_config.return_value = container_index_config

        mock_get_container_image.return_value = TestDataset.DEPLOYMENT_IMAGE_NAME
        aqua_deployment = os.path.join(
            self.curr_dir, "test_data/deployment/aqua_create_gguf_deployment.yaml"
        )
        model_deployment_obj = ModelDeployment.from_yaml(uri=aqua_deployment)
        model_deployment_dsc_obj = copy.deepcopy(
            TestDataset.model_deployment_object_gguf[0]
        )
        model_deployment_dsc_obj["lifecycle_state"] = "CREATING"
        model_deployment_obj.dsc_model_deployment = (
            oci.data_science.models.ModelDeploymentSummary(**model_deployment_dsc_obj)
        )
        mock_deploy.return_value = model_deployment_obj

        result = self.app.create(
            model_id=TestDataset.MODEL_ID,
            instance_shape="VM.Standard.A1.Flex",
            display_name="model-deployment-name",
            log_group_id="ocid1.loggroup.oc1.<region>.<OCID>",
            access_log_id="ocid1.log.oc1.<region>.<OCID>",
            predict_log_id="ocid1.log.oc1.<region>.<OCID>",
            ocpus=10.0,
            memory_in_gbs=60.0,
        )

        mock_create.assert_called_with(
            model_id=TestDataset.MODEL_ID,
            compartment_id=None,
            project_id=None,
            freeform_tags=None,
            defined_tags=None,
        )
        mock_get_container_image.assert_called()
        mock_deploy.assert_called()

        expected_attributes = set(AquaDeployment.__annotations__.keys())
        actual_attributes = asdict(result)
        assert set(actual_attributes) == set(expected_attributes), "Attributes mismatch"
        expected_result = copy.deepcopy(TestDataset.aqua_deployment_object)
        expected_result["state"] = "CREATING"
        expected_result["shape_info"] = TestDataset.aqua_deployment_gguf_shape_info
        expected_result["environment_variables"] = (
            TestDataset.aqua_deployment_gguf_env_vars
        )
        assert actual_attributes == expected_result

    @patch("ads.aqua.modeldeployment.deployment.get_container_config")
    @patch("ads.aqua.model.AquaModelApp.create")
    @patch("ads.aqua.modeldeployment.deployment.get_container_image")
    @patch("ads.model.deployment.model_deployment.ModelDeployment.deploy")
    def test_create_deployment_for_tei_byoc_embedding_model(
        self,
        mock_deploy,
        mock_get_container_image,
        mock_create,
        mock_get_container_config,
    ):
        """Test to create a deployment for fine-tuned model"""
        aqua_model = os.path.join(
            self.curr_dir, "test_data/deployment/aqua_tei_byoc_embedding_model.yaml"
        )
        datascience_model = DataScienceModel.from_yaml(uri=aqua_model)
        mock_create.return_value = datascience_model

        config_json = os.path.join(
            self.curr_dir, "test_data/deployment/deployment_config.json"
        )
        with open(config_json, "r") as _file:
            config = json.load(_file)

        self.app.get_deployment_config = MagicMock(return_value=config)

        container_index_json = os.path.join(
            self.curr_dir, "test_data/ui/container_index.json"
        )
        with open(container_index_json, "r") as _file:
            container_index_config = json.load(_file)
        mock_get_container_config.return_value = container_index_config

        mock_get_container_image.return_value = TestDataset.DEPLOYMENT_IMAGE_NAME
        aqua_deployment = os.path.join(
            self.curr_dir, "test_data/deployment/aqua_create_embedding_deployment.yaml"
        )
        model_deployment_obj = ModelDeployment.from_yaml(uri=aqua_deployment)
        model_deployment_dsc_obj = copy.deepcopy(
            TestDataset.model_deployment_object_tei_byoc[0]
        )
        model_deployment_dsc_obj["lifecycle_state"] = "CREATING"
        model_deployment_obj.dsc_model_deployment = (
            oci.data_science.models.ModelDeploymentSummary(**model_deployment_dsc_obj)
        )
        mock_deploy.return_value = model_deployment_obj

        result = self.app.create(
            model_id=TestDataset.MODEL_ID,
            instance_shape=TestDataset.DEPLOYMENT_SHAPE_NAME,
            display_name="model-deployment-name",
            log_group_id="ocid1.loggroup.oc1.<region>.<OCID>",
            access_log_id="ocid1.log.oc1.<region>.<OCID>",
            predict_log_id="ocid1.log.oc1.<region>.<OCID>",
            container_family="odsc-tei-serving",
            cmd_var=[],
        )

        mock_create.assert_called_with(
            model_id=TestDataset.MODEL_ID,
            compartment_id=None,
            project_id=None,
            freeform_tags=None,
            defined_tags=None,
        )
        mock_get_container_image.assert_called()
        mock_deploy.assert_called()

        expected_attributes = set(AquaDeployment.__annotations__.keys())
        actual_attributes = asdict(result)
        assert set(actual_attributes) == set(expected_attributes), "Attributes mismatch"
        expected_result = copy.deepcopy(TestDataset.aqua_deployment_object)
        expected_result["state"] = "CREATING"
        expected_result["shape_info"] = (
            TestDataset.aqua_deployment_tei_byoc_embeddings_shape_info
        )
        expected_result["cmd"] = TestDataset.aqua_deployment_tei_byoc_embeddings_cmd
        expected_result["environment_variables"] = (
            TestDataset.aqua_deployment_tei_byoc_embeddings_env_vars
        )
        assert actual_attributes == expected_result

    @parameterized.expand(
        [
            (
                "VLLM_PARAMS",
                "odsc-vllm-serving",
                ["--max-model-len 4096", "--seed 42", "--trust-remote-code"],
                ["--max-model-len 4096", "--trust-remote-code"],
            ),
            (
                "VLLM_PARAMS",
                "odsc-vllm-serving",
                [],
                [],
            ),
            (
                "TGI_PARAMS",
                "odsc-tgi-serving",
                ["--sharded true", "--trust-remote-code", "--max-stop-sequences"],
                ["--max-stop-sequences"],
            ),
            (
                "CUSTOM_PARAMS",
                "custom-container-key",
                ["--max-model-len 4096", "--seed 42", "--trust-remote-code"],
                ["--max-model-len 4096", "--seed 42", "--trust-remote-code"],
            ),
        ]
    )
    @patch("ads.model.datascience_model.DataScienceModel.from_id")
    def test_get_deployment_default_params(
        self,
        container_params_field,
        container_type_key,
        params,
        allowed_params,
        mock_from_id,
    ):
        """Test for fetching config details for a given deployment."""

        config_json = os.path.join(
            self.curr_dir, "test_data/deployment/deployment_config.json"
        )
        with open(config_json, "r") as _file:
            config = json.load(_file)
        # update config params for testing
        config["configuration"][TestDataset.DEPLOYMENT_SHAPE_NAME]["parameters"][
            container_params_field
        ] = " ".join(params)

        mock_model = MagicMock()
        custom_metadata_list = ModelCustomMetadata()
        custom_metadata_list.add(
            **{"key": "deployment-container", "value": container_type_key}
        )
        mock_model.custom_metadata_list = custom_metadata_list
        mock_from_id.return_value = mock_model

        self.app.get_deployment_config = MagicMock(return_value=config)
        result = self.app.get_deployment_default_params(
            TestDataset.MODEL_ID, TestDataset.DEPLOYMENT_SHAPE_NAME
        )
        if container_params_field == "CUSTOM_PARAMS":
            assert result == []
        else:
            assert result == allowed_params

    @parameterized.expand(
        [
            (
                "odsc-vllm-serving",
                ["--max-model-len 4096", "--seed 42", "--trust-remote-code"],
            ),
            (
                "odsc-vllm-serving",
                [],
            ),
            (
                "odsc-tgi-serving",
                ["--sharded true", "--trust-remote-code"],
            ),
            (
                "custom-container-key",
                ["--max-model-len 4096", "--seed 42", "--trust-remote-code"],
            ),
            (
                "odsc-vllm-serving",
                ["--port 8081"],
            ),
            (
                "odsc-tgi-serving",
                ["--port 8080"],
            ),
        ]
    )
    @patch("ads.model.datascience_model.DataScienceModel.from_id")
    @patch("ads.aqua.modeldeployment.deployment.get_container_config")
    def test_validate_deployment_params(
        self, container_type_key, params, mock_get_container_config, mock_from_id
    ):
        """Test for checking if overridden deployment params are valid."""
        mock_model = MagicMock()
        custom_metadata_list = ModelCustomMetadata()
        custom_metadata_list.add(
            **{"key": "deployment-container", "value": container_type_key}
        )
        mock_model.custom_metadata_list = custom_metadata_list
        mock_from_id.return_value = mock_model

        container_index_json = os.path.join(
            self.curr_dir, "test_data/ui/container_index.json"
        )
        with open(container_index_json, "r") as _file:
            container_index_config = json.load(_file)
        mock_get_container_config.return_value = container_index_config

        if container_type_key in {"odsc-vllm-serving", "odsc-tgi-serving"} and params:
            with pytest.raises(AquaValueError):
                self.app.validate_deployment_params(
                    model_id="mock-model-id",
                    params=params,
                )
        else:
            result = self.app.validate_deployment_params(
                model_id="mock-model-id",
                params=params,
            )
            assert result["valid"] is True

    @parameterized.expand(
        [
            (
                "odsc-vllm-serving",
                ["--max-model-len 4096"],
            ),
            (
                "odsc-tgi-serving",
                ["--max_stop_sequences 5"],
            ),
            (
                "",
                ["--some_random_key some_random_value"],
            ),
        ]
    )
    @patch("ads.model.datascience_model.DataScienceModel.from_id")
    @patch("ads.aqua.modeldeployment.deployment.get_container_config")
    def test_validate_deployment_params_for_unverified_models(
        self, container_type_key, params, mock_get_container_config, mock_from_id
    ):
        """Test to check if container family is used when metadata does not have image information
        for unverified models."""
        mock_model = MagicMock()
        mock_model.custom_metadata_list = ModelCustomMetadata()
        mock_from_id.return_value = mock_model

        container_index_json = os.path.join(
            self.curr_dir, "test_data/ui/container_index.json"
        )
        with open(container_index_json, "r") as _file:
            container_index_config = json.load(_file)
        mock_get_container_config.return_value = container_index_config

        if container_type_key in {"odsc-vllm-serving", "odsc-tgi-serving"} and params:
            result = self.app.validate_deployment_params(
                model_id="mock-model-id",
                params=params,
                container_family=container_type_key,
            )
            assert result["valid"] is True
        else:
            with pytest.raises(AquaValueError):
                self.app.validate_deployment_params(
                    model_id="mock-model-id",
                    params=params,
                    container_family=container_type_key,
                )


class TestMDInferenceResponse(unittest.TestCase):
    def setUp(self):
        self.app = MDInferenceResponse()

    @classmethod
    def setUpClass(cls):
        cls.curr_dir = os.path.dirname(os.path.abspath(__file__))

    @classmethod
    def tearDownClass(cls):
        cls.curr_dir = None

    @patch("requests.post")
    def test_get_model_deployment_response(self, mock_post):
        """Test to check if model deployment response is returned correctly."""

        endpoint = TestDataset.MODEL_DEPLOYMENT_URL + "/predict"
        self.app.prompt = "What is 1+1?"
        self.app.model_params = ModelParams(**TestDataset.model_params)

        mock_response = MagicMock()
        response_json = os.path.join(
            self.curr_dir, "test_data/deployment/aqua_deployment_response.json"
        )
        with open(response_json, "r") as _file:
            mock_response.content = _file.read()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = self.app.get_model_deployment_response(endpoint)
        assert result["choices"][0]["text"] == " The answer is 2"
