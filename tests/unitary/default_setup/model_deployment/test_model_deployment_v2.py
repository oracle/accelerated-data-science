#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
from datetime import datetime
import oci
import pytest
import unittest
import pandas
from unittest.mock import MagicMock, patch
from ads.common.oci_datascience import OCIDataScienceMixin
from ads.common.oci_logging import ConsolidatedLog, OCILog
from ads.common.oci_mixin import OCIModelMixin
from ads.model.deployment.common.utils import State
from ads.common.oci_datascience import DSCNotebookSession
from ads.model.datascience_model import DataScienceModel

from ads.model.deployment.model_deployment import (
    ModelDeployment,
    ModelDeploymentLogType,
    ModelDeploymentFailedError,
)
from ads.model.deployment.model_deployment_infrastructure import (
    ModelDeploymentInfrastructure,
)
from ads.model.deployment.model_deployment_properties import ModelDeploymentProperties
from ads.model.deployment.model_deployment_runtime import (
    ModelDeploymentContainerRuntime,
    ModelDeploymentRuntime,
)
from ads.model.service.oci_datascience_model_deployment import (
    OCIDataScienceModelDeployment,
)

try:
    from oci.data_science.models import (
        CreateModelDeploymentDetails,
        SingleModelDeploymentConfigurationDetails,
        ModelConfigurationDetails,
        InstanceConfiguration,
        ModelDeploymentInstanceShapeConfigDetails,
        FixedSizeScalingPolicy,
        # StreamConfigurationDetails,
        OcirModelDeploymentEnvironmentConfigurationDetails,
        CategoryLogDetails,
        LogDetails,
        UpdateModelDeploymentDetails,
        UpdateCategoryLogDetails,
        UpdateSingleModelDeploymentConfigurationDetails,
        UpdateOcirModelDeploymentEnvironmentConfigurationDetails,
        UpdateModelConfigurationDetails,
        # UpdateStreamConfigurationDetails,
    )
except (ImportError, AttributeError) as e:
    raise unittest.SkipTest(
        "Support for OCI ModelDeployment BYOC is not available. Skipping the ModelDeployment tests."
    )

NB_SESSION_OCID = "ocid1.datasciencenotebooksession.oc1.iad..<unique_ocid>"

OCI_MODEL_DEPLOYMENT_RESPONSE = oci.data_science.models.ModelDeployment(
    id="fakeid.datasciencemodeldeployment.oc1..xxx",
    compartment_id="fakeid.compartment.oc1..xxx",
    project_id="fakeid.datascienceproject.oc1.iad.xxx",
    display_name="Generic Model Deployment With Small Artifact",
    description="The model deployment description",
    lifecycle_state="ACTIVE",
    lifecycle_details="Model Deployment is Active.",
    created_by="fakeid.user.oc1..xxx",
    freeform_tags={"key1": "value1"},
    defined_tags={"key1": {"skey1": "value1"}},
    time_created="2022-08-24T17:07:39.200000Z",
    model_deployment_configuration_details=SingleModelDeploymentConfigurationDetails(
        deployment_type="SINGLE_MODEL",
        model_configuration_details=ModelConfigurationDetails(
            model_id="fakeid.datasciencemodel.oc1.iad.xxx",
            instance_configuration=InstanceConfiguration(
                instance_shape_name="VM.Standard.E4.Flex",
                model_deployment_instance_shape_config_details=ModelDeploymentInstanceShapeConfigDetails(
                    ocpus=10, memory_in_gbs=36
                ),
            ),
            scaling_policy=FixedSizeScalingPolicy(instance_count=5),
            bandwidth_mbps=5,
        ),
        # stream_configuration_details=StreamConfigurationDetails(
        #     input_stream_ids=["123", "456"], output_stream_ids=["321", "654"]
        # ),
        environment_configuration_details=OcirModelDeploymentEnvironmentConfigurationDetails(
            image="iad.ocir.io/ociodscdev/ml_flask_app_demo:1.0.0",
            image_digest="sha256:243590ea099af4019b6afc104b8a70b9552f0b001b37d0442f8b5a399244681c",
            entrypoint=["python", "/opt/ds/model/deployed_model/api.py"],
            cmd=[],
            server_port=5000,
            health_check_port=5000,
            environment_variables={
                "WEB_CONCURRENCY": "10",
            },
        ),
    ),
    category_log_details=CategoryLogDetails(
        access=LogDetails(
            log_id="fakeid.log.oc1.iad.xxx", log_group_id="fakeid.loggroup.oc1.iad.xxx"
        ),
        predict=LogDetails(
            log_id="fakeid.log.oc1.iad.xxx", log_group_id="fakeid.loggroup.oc1.iad.xxx"
        ),
    ),
    model_deployment_url="model_deployment_url",
    # deployment_mode="STREAM_ONLY",
)

OCI_MODEL_DEPLOYMENT_DICT = {
    "kind": "deployment",
    "type": "modelDeployment",
    "spec": {
        "display_name": "Generic Model Deployment With Small Artifact",
        "description": "The model deployment description",
        "defined_tags": {"key1": {"skey1": "value1"}},
        "freeform_tags": {"key1": "value1"},
        "infrastructure": {
            "kind": "infrastructure",
            "type": "datascienceModelDeployment",
            "spec": {
                "bandwidth_mbps": 5,
                "compartment_id": "fakeid.compartment.oc1..xxx",
                "project_id": "fakeid.datascienceproject.oc1.iad.xxx",
                "replica": 5,
                "shape_name": "VM.Standard.E4.Flex",
                "shape_config_details": {"ocpus": 10, "memoryInGBs": 36},
                "web_concurrency": 10,
                "access_log": {
                    "logGroupId": "fakeid.loggroup.oc1.iad.xxx",
                    "logId": "fakeid.log.oc1.iad.xxx",
                },
                "predict_log": {
                    "logGroupId": "fakeid.loggroup.oc1.iad.xxx",
                    "logId": "fakeid.log.oc1.iad.xxx",
                },
            },
        },
        "runtime": {
            "kind": "runtime",
            "type": "container",
            "spec": {
                "image": "iad.ocir.io/ociodscdev/ml_flask_app_demo:1.0.0",
                "image_digest": "sha256:243590ea099af4019b6afc104b8a70b9552f0b001b37d0442f8b5a399244681c",
                "entrypoint": ["python", "/opt/ds/model/deployed_model/api.py"],
                "server_port": 5000,
                "health_check_port": 5000,
                "env": {"WEB_CONCURRENCY": "10"},
                # "input_stream_ids": ["123", "456"],
                # "output_stream_ids": ["321", "654"],
                "model_uri": "fakeid.datasciencemodel.oc1.iad.xxx",
                "deployment_mode": "HTTPS_ONLY",
            },
        },
    },
}

MODEL_DEPLOYMENT_YAML = """
kind: deployment
spec:
  displayName: Generic Model Deployment With Small Artifact
  description: The model deployment description
  freeform_tags:
    key1: value1
  defined_tags:
    key1:
      skey1: value1
  infrastructure:
    kind: infrastructure
    type: datascienceModelDeployment
    spec:
      compartmentId: fakeid.compartment.oc1..xxx
      projectId: fakeid.datascienceproject.oc1.iad.xxx
      accessLog:
        logGroupId: fakeid.loggroup.oc1.iad.xxx
        logId: fakeid.log.oc1.iad.xxx
      predictLog:
        logGroupId: fakeid.loggroup.oc1.iad.xxx
        logId: fakeid.log.oc1.iad.xxx
      shapeName: VM.Standard.E4.Flex
      shapeConfigDetails:
        memoryInGBs: 36
        ocpus: 10
      replica: 5
      bandwidthMbps: 5
      webConcurrency: 10
  runtime:
    kind: runtime
    type: container
    spec:
      modelUri: fakeid.datasciencemodel.oc1.iad.xxx
      image: iad.ocir.io/ociodscdev/ml_flask_app_demo:1.0.0
      imageDigest: sha256:243590ea099af4019b6afc104b8a70b9552f0b001b37d0442f8b5a399244681c
      entrypoint: ["python", "/opt/ds/model/deployed_model/api.py"]
      serverPort: 5000
      healthCheckPort: 5000
      env:
        key: value
      deploymentMode: HTTPS_ONLY
"""

infrastructure = (
    ModelDeploymentInfrastructure()
    .with_bandwidth_mbps(5)
    .with_compartment_id("fakeid.compartment.oc1..xxx")
    .with_project_id("fakeid.datascienceproject.oc1.iad.xxx")
    .with_replica(5)
    .with_shape_name("VM.Standard.E4.Flex")
    .with_shape_config_details(ocpus=10, memory_in_gbs=36)
    .with_web_concurrency(10)
    .with_access_log(
        log_group_id="fakeid.loggroup.oc1.iad.xxx",
        log_id="fakeid.log.oc1.iad.xxx",
    )
    .with_predict_log(
        log_group_id="fakeid.loggroup.oc1.iad.xxx",
        log_id="fakeid.log.oc1.iad.xxx",
    )
)

runtime = (
    ModelDeploymentContainerRuntime()
    .with_image("iad.ocir.io/ociodscdev/ml_flask_app_demo:1.0.0")
    .with_image_digest(
        "sha256:243590ea099af4019b6afc104b8a70b9552f0b001b37d0442f8b5a399244681c"
    )
    .with_entrypoint(["python", "/opt/ds/model/deployed_model/api.py"])
    .with_server_port(5000)
    .with_health_check_port(5000)
    # .with_input_stream_ids(["123", "456"])
    # .with_output_stream_ids(["321", "654"])
    .with_model_uri("fakeid.datasciencemodel.oc1.iad.xxx")
    .with_deployment_mode("HTTPS_ONLY")
)

nb_session = DSCNotebookSession(
    **{
        "notebook_session_configuration_details": {
            "shape": "VM.Standard.E4.Flex",
            "block_storage_size_in_gbs": 100,
            "subnet_id": "test_subnet_id",
            "notebook_session_shape_config_details": {
                "ocpus": 10.0,
                "memory_in_gbs": 36.0,
            },
        }
    }
)


class ModelDeploymentBYOCTestCase(unittest.TestCase):
    def initialize_model_deployment(self):
        model_deployment = (
            ModelDeployment()
            .with_display_name("Generic Model Deployment With Small Artifact")
            .with_description("The model deployment description")
            .with_defined_tags(key1={"skey1": "value1"})
            .with_freeform_tags(key1="value1")
            .with_infrastructure(infrastructure)
            .with_runtime(runtime)
        )

        return model_deployment

    def initialize_model_deployment_from_spec(self):
        return ModelDeployment(
            spec={
                "display_name": "Generic Model Deployment With Small Artifact",
                "description": "The model deployment description",
                "defined_tags": {"key1": {"skey1": "value1"}},
                "freeform_tags": {"key1": "value1"},
                "infrastructure": infrastructure,
                "runtime": runtime,
            }
        )

    def initialize_model_deployment_triton_builder(self):
        infrastructure = (
            ModelDeploymentInfrastructure()
            .with_compartment_id("fakeid.compartment.oc1..xxx")
            .with_project_id("fakeid.datascienceproject.oc1.iad.xxx")
            .with_shape_name("VM.Standard.E4.Flex")
            .with_replica(2)
            .with_bandwidth_mbps(10)
        )
        runtime = (
            ModelDeploymentContainerRuntime()
            .with_image("fake_image")
            .with_server_port(5000)
            .with_health_check_port(5000)
            .with_model_uri("fake_model_id")
            .with_env({"key": "value", "key2": "value2"})
            .with_inference_server("triton")
        )

        deployment = (
            ModelDeployment()
            .with_display_name("triton case")
            .with_infrastructure(infrastructure)
            .with_runtime(runtime)
        )
        return deployment

    def initialize_model_deployment_triton_yaml(self):
        yaml_string = """
kind: deployment
spec:
  displayName: triton
  infrastructure:
    kind: infrastructure
    spec:
      bandwidthMbps: 10
      compartmentId: fake_compartment_id
      deploymentType: SINGLE_MODEL
      policyType: FIXED_SIZE
      replica: 2
      shapeConfigDetails:
        memoryInGBs: 16.0
        ocpus: 1.0
      shapeName: VM.Standard.E4.Flex
    type: datascienceModelDeployment
  runtime:
    kind: runtime
    spec:
      env:
        key: value
        key2: value2
      inference_server: triton
      healthCheckPort: 8000
      image: fake_image
      modelUri: fake_model_id
      serverPort: 8000
    type: container
"""
        deployment_from_yaml = ModelDeployment.from_yaml(yaml_string)
        return deployment_from_yaml

    def initialize_model_deployment_from_kwargs(self):
        return ModelDeployment(
            display_name="Generic Model Deployment With Small Artifact",
            description="The model deployment description",
            defined_tags={"key1": {"skey1": "value1"}},
            freeform_tags={"key1": "value1"},
            infrastructure=infrastructure,
            runtime=runtime,
        )

    @patch(
        "ads.model.deployment.model_deployment_infrastructure.COMPARTMENT_OCID",
        infrastructure.compartment_id,
    )
    @patch(
        "ads.model.deployment.model_deployment_infrastructure.PROJECT_OCID",
        infrastructure.project_id,
    )
    @patch(
        "ads.model.deployment.model_deployment_infrastructure.NB_SESSION_OCID",
        NB_SESSION_OCID,
    )
    @patch.object(DSCNotebookSession, "from_ocid")
    def test__load_default_properties(self, mock_from_ocid):
        mock_default_properties = {
            ModelDeploymentInfrastructure.CONST_COMPARTMENT_ID: infrastructure.compartment_id,
            ModelDeploymentInfrastructure.CONST_PROJECT_ID: infrastructure.project_id,
            ModelDeploymentInfrastructure.CONST_SHAPE_NAME: infrastructure.shape_name,
            ModelDeploymentInfrastructure.CONST_BANDWIDTH_MBPS: 10,
            ModelDeploymentInfrastructure.CONST_SHAPE_CONFIG_DETAILS: {
                "ocpus": 10.0,
                "memory_in_gbs": 36.0,
            },
            ModelDeploymentInfrastructure.CONST_WEB_CONCURRENCY: 10,
            ModelDeploymentInfrastructure.CONST_REPLICA: 1,
        }

        mock_from_ocid.return_value = nb_session
        assert infrastructure._load_default_properties() == mock_default_properties
        mock_from_ocid.assert_called_with(NB_SESSION_OCID)

    def test_initialize_model_deployment(self):
        temp_model_deployment = self.initialize_model_deployment()

        assert (
            temp_model_deployment.display_name
            == "Generic Model Deployment With Small Artifact"
        )
        assert temp_model_deployment.description == "The model deployment description"
        assert temp_model_deployment.defined_tags == {"key1": {"skey1": "value1"}}
        assert temp_model_deployment.freeform_tags == {"key1": "value1"}
        assert isinstance(
            temp_model_deployment.runtime, ModelDeploymentContainerRuntime
        )
        assert isinstance(
            temp_model_deployment.infrastructure, ModelDeploymentInfrastructure
        )

        temp_runtime = temp_model_deployment.runtime
        assert temp_runtime.environment_config_type == "OCIR_CONTAINER"
        assert temp_runtime.env == {"WEB_CONCURRENCY": "10"}
        assert temp_runtime.deployment_mode == "HTTPS_ONLY"
        # assert temp_runtime.input_stream_ids == ["123", "456"]
        # assert temp_runtime.output_stream_ids == ["321", "654"]
        assert temp_runtime.image == "iad.ocir.io/ociodscdev/ml_flask_app_demo:1.0.0"
        assert (
            temp_runtime.image_digest
            == "sha256:243590ea099af4019b6afc104b8a70b9552f0b001b37d0442f8b5a399244681c"
        )
        assert temp_runtime.entrypoint == [
            "python",
            "/opt/ds/model/deployed_model/api.py",
        ]
        assert temp_runtime.server_port == 5000
        assert temp_runtime.health_check_port == 5000
        assert temp_runtime.model_uri == "fakeid.datasciencemodel.oc1.iad.xxx"

        temp_infrastructure = temp_model_deployment.infrastructure
        assert temp_infrastructure.bandwidth_mbps == 5
        assert temp_infrastructure.compartment_id == "fakeid.compartment.oc1..xxx"
        assert temp_infrastructure.project_id == "fakeid.datascienceproject.oc1.iad.xxx"
        assert temp_infrastructure.web_concurrency == 10
        assert temp_infrastructure.shape_name == "VM.Standard.E4.Flex"
        assert temp_infrastructure.shape_config_details == {
            "ocpus": 10,
            "memoryInGBs": 36,
        }
        assert temp_infrastructure.replica == 5
        assert temp_infrastructure.access_log == {
            "logGroupId": "fakeid.loggroup.oc1.iad.xxx",
            "logId": "fakeid.log.oc1.iad.xxx",
        }
        assert temp_infrastructure.predict_log == {
            "logGroupId": "fakeid.loggroup.oc1.iad.xxx",
            "logId": "fakeid.log.oc1.iad.xxx",
        }

    def test_initialize_model_deployment_from_spec(self):
        model_deployment_spec = self.initialize_model_deployment_from_spec()
        model_deployment_builder = self.initialize_model_deployment()

        assert model_deployment_spec.to_dict() == model_deployment_builder.to_dict()

    def test_initialize_model_deployment_with_error(self):
        with pytest.raises(
            ValueError,
            match="You can only pass in either `spec` or `properties` to initialize model deployment instance.",
        ):
            model_deployment = ModelDeployment(
                properties=ModelDeploymentProperties(),
                spec={
                    "display_name": "Generic Model Deployment With Small Artifact",
                    "description": "The model deployment description",
                    "defined_tags": {"key1": {"skey1": "value1"}},
                    "freeform_tags": {"key1": "value1"},
                },
            )

    def test_initialize_model_deployment_with_spec_kwargs(self):
        model_deployment_kwargs = self.initialize_model_deployment_from_kwargs()
        model_deployment_builder = self.initialize_model_deployment()

        assert model_deployment_kwargs.to_dict() == model_deployment_builder.to_dict()

    def test_initialize_model_deployment_triton_builder(self):
        temp_model_deployment = self.initialize_model_deployment_triton_builder()
        assert isinstance(
            temp_model_deployment.runtime, ModelDeploymentContainerRuntime
        )
        assert isinstance(
            temp_model_deployment.infrastructure, ModelDeploymentInfrastructure
        )
        assert temp_model_deployment.runtime.inference_server == "triton"

    def test_initialize_model_deployment_triton_yaml(self):
        temp_model_deployment = self.initialize_model_deployment_triton_yaml()
        assert isinstance(
            temp_model_deployment.runtime, ModelDeploymentContainerRuntime
        )
        assert isinstance(
            temp_model_deployment.infrastructure, ModelDeploymentInfrastructure
        )
        assert temp_model_deployment.runtime.inference_server == "triton"

    def test_model_deployment_to_dict(self):
        model_deployment = self.initialize_model_deployment()
        assert model_deployment.to_dict() == {
            "kind": "deployment",
            "type": "modelDeployment",
            "spec": {
                "displayName": "Generic Model Deployment With Small Artifact",
                "description": "The model deployment description",
                "definedTags": {"key1": {"skey1": "value1"}},
                "freeformTags": {"key1": "value1"},
                "infrastructure": {
                    "kind": "infrastructure",
                    "type": "datascienceModelDeployment",
                    "spec": {
                        "bandwidthMbps": 5,
                        "compartmentId": "fakeid.compartment.oc1..xxx",
                        "projectId": "fakeid.datascienceproject.oc1.iad.xxx",
                        "replica": 5,
                        "shapeName": "VM.Standard.E4.Flex",
                        "shapeConfigDetails": {"ocpus": 10, "memoryInGBs": 36},
                        "webConcurrency": 10,
                        "accessLog": {
                            "logGroupId": "fakeid.loggroup.oc1.iad.xxx",
                            "logId": "fakeid.log.oc1.iad.xxx",
                        },
                        "predictLog": {
                            "logGroupId": "fakeid.loggroup.oc1.iad.xxx",
                            "logId": "fakeid.log.oc1.iad.xxx",
                        },
                    },
                },
                "runtime": {
                    "kind": "runtime",
                    "type": "container",
                    "spec": {
                        "image": "iad.ocir.io/ociodscdev/ml_flask_app_demo:1.0.0",
                        "imageDigest": "sha256:243590ea099af4019b6afc104b8a70b9552f0b001b37d0442f8b5a399244681c",
                        "entrypoint": ["python", "/opt/ds/model/deployed_model/api.py"],
                        "serverPort": 5000,
                        "healthCheckPort": 5000,
                        "env": {"WEB_CONCURRENCY": "10"},
                        # "inputStreamIds": ["123", "456"],
                        # "outputStreamIds": ["321", "654"],
                        "modelUri": "fakeid.datasciencemodel.oc1.iad.xxx",
                        "deploymentMode": "HTTPS_ONLY",
                    },
                },
            },
        }

    @patch.object(DataScienceModel, "create")
    def test_build_model_deployment_configuration_details(self, mock_create):
        dsc_model = MagicMock()
        dsc_model.id = "fakeid.datasciencemodel.oc1.iad.xxx"
        mock_create.return_value = dsc_model
        model_deployment = self.initialize_model_deployment()
        model_deployment_configuration_details = (
            model_deployment._build_model_deployment_configuration_details()
        )

        mock_create.assert_called()
        assert model_deployment_configuration_details == {
            "deploymentType": "SINGLE_MODEL",
            "modelConfigurationDetails": {
                "bandwidthMbps": 5,
                "instanceConfiguration": {
                    "instanceShapeName": "VM.Standard.E4.Flex",
                    "modelDeploymentInstanceShapeConfigDetails": {
                        "ocpus": 10,
                        "memoryInGBs": 36,
                    },
                },
                "modelId": "fakeid.datasciencemodel.oc1.iad.xxx",
                "scalingPolicy": {"policyType": "FIXED_SIZE", "instanceCount": 5},
            },
            # "streamConfigurationDetails": {
            #     "inputStreamIds": ["123", "456"],
            #     "outputStreamIds": ["321", "654"],
            # },
            "environmentConfigurationDetails": {
                "environmentConfigurationType": "OCIR_CONTAINER",
                "image": "iad.ocir.io/ociodscdev/ml_flask_app_demo:1.0.0",
                "imageDigest": "sha256:243590ea099af4019b6afc104b8a70b9552f0b001b37d0442f8b5a399244681c",
                "entrypoint": ["python", "/opt/ds/model/deployed_model/api.py"],
                "cmd": [],
                "serverPort": 5000,
                "healthCheckPort": 5000,
                "environmentVariables": {"WEB_CONCURRENCY": "10"},
            },
        }

    def test_build_category_log_details(self):
        model_deployment = self.initialize_model_deployment()
        category_log_details = model_deployment._build_category_log_details()

        assert category_log_details == {
            "access": {
                "logGroupId": "fakeid.loggroup.oc1.iad.xxx",
                "logId": "fakeid.log.oc1.iad.xxx",
            },
            "predict": {
                "logGroupId": "fakeid.loggroup.oc1.iad.xxx",
                "logId": "fakeid.log.oc1.iad.xxx",
            },
        }

    @patch.object(DataScienceModel, "create")
    def test_build_model_deployment_details(self, mock_create):
        dsc_model = MagicMock()
        dsc_model.id = "fakeid.datasciencemodel.oc1.iad.xxx"
        mock_create.return_value = dsc_model
        model_deployment = self.initialize_model_deployment()
        create_model_deployment_details = (
            model_deployment._build_model_deployment_details()
        )

        mock_create.assert_called()

        assert isinstance(
            create_model_deployment_details,
            CreateModelDeploymentDetails,
        )
        assert (
            create_model_deployment_details.display_name
            == model_deployment.display_name
        )
        assert (
            create_model_deployment_details.description == model_deployment.description
        )
        assert (
            create_model_deployment_details.freeform_tags
            == model_deployment.freeform_tags
        )
        assert (
            create_model_deployment_details.defined_tags
            == model_deployment.defined_tags
        )

        category_log_details = create_model_deployment_details.category_log_details
        assert isinstance(category_log_details, CategoryLogDetails)
        assert (
            category_log_details.access.log_id
            == model_deployment.infrastructure.access_log["logId"]
        )
        assert (
            category_log_details.access.log_group_id
            == model_deployment.infrastructure.access_log["logGroupId"]
        )
        assert (
            category_log_details.predict.log_id
            == model_deployment.infrastructure.predict_log["logId"]
        )
        assert (
            category_log_details.predict.log_group_id
            == model_deployment.infrastructure.predict_log["logGroupId"]
        )

        model_deployment_configuration_details = (
            create_model_deployment_details.model_deployment_configuration_details
        )
        assert isinstance(
            model_deployment_configuration_details,
            SingleModelDeploymentConfigurationDetails,
        )
        assert model_deployment_configuration_details.deployment_type == "SINGLE_MODEL"

        environment_configuration_details = (
            model_deployment_configuration_details.environment_configuration_details
        )
        assert isinstance(
            environment_configuration_details,
            OcirModelDeploymentEnvironmentConfigurationDetails,
        )
        assert (
            environment_configuration_details.environment_configuration_type
            == "OCIR_CONTAINER"
        )
        assert (
            environment_configuration_details.environment_variables
            == model_deployment.runtime.env
        )
        assert environment_configuration_details.cmd == model_deployment.runtime.cmd
        assert environment_configuration_details.image == model_deployment.runtime.image
        assert (
            environment_configuration_details.image_digest
            == model_deployment.runtime.image_digest
        )
        assert (
            environment_configuration_details.entrypoint
            == model_deployment.runtime.entrypoint
        )
        assert (
            environment_configuration_details.server_port
            == model_deployment.runtime.server_port
        )
        assert (
            environment_configuration_details.health_check_port
            == model_deployment.runtime.health_check_port
        )

        model_configuration_details = (
            model_deployment_configuration_details.model_configuration_details
        )
        assert isinstance(
            model_configuration_details,
            ModelConfigurationDetails,
        )
        assert (
            model_configuration_details.bandwidth_mbps
            == model_deployment.infrastructure.bandwidth_mbps
        )
        assert (
            model_configuration_details.model_id == model_deployment.runtime.model_uri
        )

        instance_configuration = model_configuration_details.instance_configuration
        assert isinstance(instance_configuration, InstanceConfiguration)
        assert (
            instance_configuration.instance_shape_name
            == model_deployment.infrastructure.shape_name
        )
        assert (
            instance_configuration.model_deployment_instance_shape_config_details.ocpus
            == model_deployment.infrastructure.shape_config_details["ocpus"]
        )
        assert (
            instance_configuration.model_deployment_instance_shape_config_details.memory_in_gbs
            == model_deployment.infrastructure.shape_config_details["memoryInGBs"]
        )

        scaling_policy = model_configuration_details.scaling_policy
        assert isinstance(scaling_policy, FixedSizeScalingPolicy)
        assert scaling_policy.policy_type == "FIXED_SIZE"
        assert scaling_policy.instance_count == model_deployment.infrastructure.replica

        # stream_configuration_details = (
        #     model_deployment_configuration_details.stream_configuration_details
        # )
        # assert isinstance(
        #     stream_configuration_details,
        #     StreamConfigurationDetails,
        # )
        # assert (
        #     stream_configuration_details.input_stream_ids
        #     == model_deployment.runtime.input_stream_ids
        # )
        # assert (
        #     stream_configuration_details.output_stream_ids
        #     == model_deployment.runtime.output_stream_ids
        # )

    def test_update_from_oci_model(self):
        model_deployment = self.initialize_model_deployment()
        model_deployment_from_oci = model_deployment._update_from_oci_model(
            OCI_MODEL_DEPLOYMENT_RESPONSE
        )

        assert isinstance(model_deployment_from_oci, ModelDeployment)
        assert (
            model_deployment_from_oci.model_deployment_id
            == OCI_MODEL_DEPLOYMENT_RESPONSE.id
        )
        assert (
            model_deployment_from_oci.display_name
            == OCI_MODEL_DEPLOYMENT_RESPONSE.display_name
        )
        assert (
            model_deployment_from_oci.defined_tags
            == OCI_MODEL_DEPLOYMENT_RESPONSE.defined_tags
        )
        assert (
            model_deployment_from_oci.freeform_tags
            == OCI_MODEL_DEPLOYMENT_RESPONSE.freeform_tags
        )
        assert (
            model_deployment_from_oci.description
            == OCI_MODEL_DEPLOYMENT_RESPONSE.description
        )
        assert (
            model_deployment_from_oci.lifecycle_state
            == OCI_MODEL_DEPLOYMENT_RESPONSE.lifecycle_state
        )
        assert (
            model_deployment_from_oci.lifecycle_details
            == OCI_MODEL_DEPLOYMENT_RESPONSE.lifecycle_details
        )
        assert (
            model_deployment_from_oci.created_by
            == OCI_MODEL_DEPLOYMENT_RESPONSE.created_by
        )
        assert (
            model_deployment_from_oci.time_created
            == OCI_MODEL_DEPLOYMENT_RESPONSE.time_created
        )
        assert (
            model_deployment_from_oci.url
            == OCI_MODEL_DEPLOYMENT_RESPONSE.model_deployment_url
        )

        infrastructure = model_deployment_from_oci.infrastructure

        assert isinstance(infrastructure, ModelDeploymentInfrastructure)
        assert (
            infrastructure.compartment_id
            == OCI_MODEL_DEPLOYMENT_RESPONSE.compartment_id
        )
        assert infrastructure.project_id == OCI_MODEL_DEPLOYMENT_RESPONSE.project_id

        model_deployment_configuration_details = (
            OCI_MODEL_DEPLOYMENT_RESPONSE.model_deployment_configuration_details
        )
        model_configuration_details = (
            model_deployment_configuration_details.model_configuration_details
        )
        instance_configuration = model_configuration_details.instance_configuration
        scaling_policy = model_configuration_details.scaling_policy

        assert (
            infrastructure.bandwidth_mbps == model_configuration_details.bandwidth_mbps
        )
        assert infrastructure.shape_name == instance_configuration.instance_shape_name
        assert (
            infrastructure.shape_config_details["ocpus"]
            == instance_configuration.model_deployment_instance_shape_config_details.ocpus
        )
        assert (
            infrastructure.shape_config_details["memoryInGBs"]
            == instance_configuration.model_deployment_instance_shape_config_details.memory_in_gbs
        )
        assert infrastructure.replica == scaling_policy.instance_count

        category_log_details = OCI_MODEL_DEPLOYMENT_RESPONSE.category_log_details
        assert infrastructure.access_log["logId"] == category_log_details.access.log_id
        assert (
            infrastructure.access_log["logGroupId"]
            == category_log_details.access.log_group_id
        )
        assert (
            infrastructure.predict_log["logId"] == category_log_details.predict.log_id
        )
        assert (
            infrastructure.predict_log["logGroupId"]
            == category_log_details.predict.log_group_id
        )

        runtime = model_deployment_from_oci.runtime
        assert isinstance(runtime, ModelDeploymentContainerRuntime)

        environment_configuration_details = (
            model_deployment_configuration_details.environment_configuration_details
        )
        #         stream_configuration_details = (
        #             model_deployment_configuration_details.stream_configuration_details
        #         )
        assert (
            runtime.environment_config_type
            == environment_configuration_details.environment_configuration_type
        )
        assert runtime.env == environment_configuration_details.environment_variables
        assert runtime.image == environment_configuration_details.image
        assert runtime.image_digest == environment_configuration_details.image_digest
        assert runtime.entrypoint == environment_configuration_details.entrypoint
        assert runtime.cmd == environment_configuration_details.cmd
        assert runtime.server_port == environment_configuration_details.server_port
        assert (
            runtime.health_check_port
            == environment_configuration_details.health_check_port
        )
        # assert runtime.input_stream_ids == stream_configuration_details.input_stream_ids
        # assert (
        #     runtime.output_stream_ids == stream_configuration_details.output_stream_ids
        # )
        # assert runtime.deployment_mode == OCI_MODEL_DEPLOYMENT_RESPONSE.deployment_mode
        assert runtime.model_uri == model_configuration_details.model_id
        assert (
            infrastructure.web_concurrency
            == environment_configuration_details.environment_variables[
                "WEB_CONCURRENCY"
            ]
        )

    def test_model_deployment_from_yaml(self):
        model_deployment_from_yaml = ModelDeployment.from_yaml(
            yaml_string=MODEL_DEPLOYMENT_YAML
        )
        assert isinstance(model_deployment_from_yaml, ModelDeployment)
        assert isinstance(
            model_deployment_from_yaml.infrastructure, ModelDeploymentInfrastructure
        )
        assert isinstance(model_deployment_from_yaml.runtime, ModelDeploymentRuntime)

    def test_model_deployment_from_dict(self):
        new_model_deployment = ModelDeployment.from_dict(
            copy.deepcopy(OCI_MODEL_DEPLOYMENT_DICT)
        )
        model_deployment = self.initialize_model_deployment()

        assert new_model_deployment.to_dict() == model_deployment.to_dict()

    @patch.object(DataScienceModel, "create")
    def test_update_model_deployment_details(self, mock_create):
        dsc_model = MagicMock()
        dsc_model.id = "fakeid.datasciencemodel.oc1.iad.xxx"
        mock_create.return_value = dsc_model        
        model_deployment = self.initialize_model_deployment()
        update_model_deployment_details = (
            model_deployment._update_model_deployment_details()
        )

        mock_create.assert_called()

        assert isinstance(
            update_model_deployment_details,
            UpdateModelDeploymentDetails,
        )
        assert (
            update_model_deployment_details.display_name
            == model_deployment.display_name
        )
        assert (
            update_model_deployment_details.description == model_deployment.description
        )
        assert (
            update_model_deployment_details.freeform_tags
            == model_deployment.freeform_tags
        )
        assert (
            update_model_deployment_details.defined_tags
            == model_deployment.defined_tags
        )

        category_log_details = update_model_deployment_details.category_log_details
        assert isinstance(category_log_details, UpdateCategoryLogDetails)
        assert (
            category_log_details.access.log_id
            == model_deployment.infrastructure.access_log["logId"]
        )
        assert (
            category_log_details.access.log_group_id
            == model_deployment.infrastructure.access_log["logGroupId"]
        )
        assert (
            category_log_details.predict.log_id
            == model_deployment.infrastructure.predict_log["logId"]
        )
        assert (
            category_log_details.predict.log_group_id
            == model_deployment.infrastructure.predict_log["logGroupId"]
        )

        model_deployment_configuration_details = (
            update_model_deployment_details.model_deployment_configuration_details
        )
        assert isinstance(
            model_deployment_configuration_details,
            UpdateSingleModelDeploymentConfigurationDetails,
        )
        assert model_deployment_configuration_details.deployment_type == "SINGLE_MODEL"

        environment_configuration_details = (
            model_deployment_configuration_details.environment_configuration_details
        )
        assert isinstance(
            environment_configuration_details,
            UpdateOcirModelDeploymentEnvironmentConfigurationDetails,
        )
        assert (
            environment_configuration_details.environment_configuration_type
            == "OCIR_CONTAINER"
        )
        assert (
            environment_configuration_details.environment_variables
            == model_deployment.runtime.env
        )
        assert environment_configuration_details.cmd == model_deployment.runtime.cmd
        assert environment_configuration_details.image == model_deployment.runtime.image
        assert (
            environment_configuration_details.image_digest
            == model_deployment.runtime.image_digest
        )
        assert (
            environment_configuration_details.entrypoint
            == model_deployment.runtime.entrypoint
        )
        assert (
            environment_configuration_details.server_port
            == model_deployment.runtime.server_port
        )
        assert (
            environment_configuration_details.health_check_port
            == model_deployment.runtime.health_check_port
        )

        model_configuration_details = (
            model_deployment_configuration_details.model_configuration_details
        )
        assert isinstance(
            model_configuration_details,
            UpdateModelConfigurationDetails,
        )
        assert (
            model_configuration_details.bandwidth_mbps
            == model_deployment.infrastructure.bandwidth_mbps
        )
        assert (
            model_configuration_details.model_id == model_deployment.runtime.model_uri
        )

        instance_configuration = model_configuration_details.instance_configuration
        assert isinstance(instance_configuration, InstanceConfiguration)
        assert (
            instance_configuration.instance_shape_name
            == model_deployment.infrastructure.shape_name
        )
        assert (
            instance_configuration.model_deployment_instance_shape_config_details.ocpus
            == model_deployment.infrastructure.shape_config_details["ocpus"]
        )
        assert (
            instance_configuration.model_deployment_instance_shape_config_details.memory_in_gbs
            == model_deployment.infrastructure.shape_config_details["memoryInGBs"]
        )

        scaling_policy = model_configuration_details.scaling_policy
        assert isinstance(scaling_policy, FixedSizeScalingPolicy)
        assert scaling_policy.policy_type == "FIXED_SIZE"
        assert scaling_policy.instance_count == model_deployment.infrastructure.replica

        # stream_configuration_details = (
        #     model_deployment_configuration_details.stream_configuration_details
        # )
        # assert isinstance(
        #     stream_configuration_details,
        #     UpdateStreamConfigurationDetails,
        # )
        # assert (
        #     stream_configuration_details.input_stream_ids
        #     == model_deployment.runtime.input_stream_ids
        # )
        # assert (
        #     stream_configuration_details.output_stream_ids
        #     == model_deployment.runtime.output_stream_ids
        # )

    @patch.object(
        ModelDeploymentInfrastructure, "_load_default_properties", return_value={}
    )
    def test_extract_from_oci_model(self, mock_load_default_properties):
        infrastructure = ModelDeploymentInfrastructure()
        runtime = ModelDeploymentContainerRuntime()

        assert infrastructure.to_dict() == {
            "kind": "infrastructure",
            "type": "datascienceModelDeployment",
            "spec": {},
        }

        assert runtime.to_dict() == {"kind": "runtime", "type": "container", "spec": {}}

        ModelDeployment._extract_from_oci_model(
            infrastructure,
            OCI_MODEL_DEPLOYMENT_RESPONSE,
            infrastructure.sub_level_attribute_maps,
        )
        ModelDeployment._extract_from_oci_model(runtime, OCI_MODEL_DEPLOYMENT_RESPONSE)

        assert infrastructure.to_dict() == {
            "kind": "infrastructure",
            "type": "datascienceModelDeployment",
            "spec": {
                "bandwidthMbps": 5,
                "compartmentId": "fakeid.compartment.oc1..xxx",
                "projectId": "fakeid.datascienceproject.oc1.iad.xxx",
                "replica": 5,
                "shapeName": "VM.Standard.E4.Flex",
                "shapeConfigDetails": {"ocpus": 10, "memoryInGBs": 36},
                "accessLog": {
                    "logGroupId": "fakeid.loggroup.oc1.iad.xxx",
                    "logId": "fakeid.log.oc1.iad.xxx",
                },
                "predictLog": {
                    "logGroupId": "fakeid.loggroup.oc1.iad.xxx",
                    "logId": "fakeid.log.oc1.iad.xxx",
                },
                "deploymentType": "SINGLE_MODEL",
                "policyType": "FIXED_SIZE",
            },
        }

        assert runtime.to_dict() == {
            "kind": "runtime",
            "type": "container",
            "spec": {
                "image": "iad.ocir.io/ociodscdev/ml_flask_app_demo:1.0.0",
                "imageDigest": "sha256:243590ea099af4019b6afc104b8a70b9552f0b001b37d0442f8b5a399244681c",
                "entrypoint": ["python", "/opt/ds/model/deployed_model/api.py"],
                "serverPort": 5000,
                "healthCheckPort": 5000,
                "env": {"WEB_CONCURRENCY": "10"},
                # "inputStreamIds": ["123", "456"],
                # "outputStreamIds": ["321", "654"],
                "modelUri": "fakeid.datasciencemodel.oc1.iad.xxx",
                # "deploymentMode": "HTTPS_ONLY",
            },
        }

    def test_extract_spec_kwargs(self):
        kwargs = {
            "display_name": "name",
            "description": "description",
            "freeform_tags": {"key": "value"},
            "defined_tags": {"key": "value"},
            "infrastructure": "infrastructure",
            "runtime": "runtime",
            "A": "A",
            "B": "B",
        }
        spec_kwargs = ModelDeployment()._extract_spec_kwargs(**kwargs)

        assert spec_kwargs == {
            "display_name": "name",
            "description": "description",
            "freeform_tags": {"key": "value"},
            "defined_tags": {"key": "value"},
            "infrastructure": "infrastructure",
            "runtime": "runtime",
        }

    @patch.object(OCIDataScienceMixin, "from_ocid")
    def test_from_ocid(self, mock_from_ocid):
        mock_from_ocid.return_value = (
            OCIDataScienceModelDeployment().update_from_oci_model(
                OCI_MODEL_DEPLOYMENT_RESPONSE
            )
        )
        ModelDeployment.from_id("test_ocid")
        mock_from_ocid.assert_called_with("test_ocid")

    @patch.object(OCIDataScienceMixin, "sync")
    @patch.object(
        oci.data_science.DataScienceClient,
        "create_model_deployment",
    )
    @patch.object(DataScienceModel, "create")
    def test_deploy(
        self, mock_create, mock_create_model_deployment, mock_sync
    ):
        dsc_model = MagicMock()
        dsc_model.id = "fakeid.datasciencemodel.oc1.iad.xxx"
        mock_create.return_value = dsc_model
        response = MagicMock()
        response.data = OCI_MODEL_DEPLOYMENT_RESPONSE
        mock_create_model_deployment.return_value = response
        model_deployment = self.initialize_model_deployment()
        model_deployment.set_spec(model_deployment.CONST_ID, "test_model_deployment_id")
        create_model_deployment_details = (
            model_deployment._build_model_deployment_details()
        )
        model_deployment.deploy(wait_for_completion=False)
        mock_create.assert_called()
        mock_create_model_deployment.assert_called_with(create_model_deployment_details)
        mock_sync.assert_called()

    @patch.object(OCIDataScienceMixin, "sync")
    @patch.object(
        oci.data_science.DataScienceClient,
        "create_model_deployment",
    )
    @patch.object(DataScienceModel, "create")
    def test_deploy_failed(
        self, mock_create, mock_create_model_deployment, mock_sync
    ):
        dsc_model = MagicMock()
        dsc_model.id = "fakeid.datasciencemodel.oc1.iad.xxx"
        mock_create.return_value = dsc_model
        response = oci.response.Response(
            status=MagicMock(),
            headers=MagicMock(),
            request=MagicMock(),
            data=oci.data_science.models.ModelDeployment(
                id="test_model_deployment_id",
                lifecycle_state="FAILED",
                lifecycle_details="The specified log object is not found or user is not authorized.",
            ),
        )
        mock_sync.return_value = response.data
        model_deployment = self.initialize_model_deployment()
        create_model_deployment_details = (
            model_deployment._build_model_deployment_details()
        )
        with pytest.raises(
            ModelDeploymentFailedError,
            match=f"Model deployment {response.data.id} failed to deploy: {response.data.lifecycle_details}",
        ):
            model_deployment.deploy(wait_for_completion=False)
            mock_create.assert_called()
            mock_create_model_deployment.assert_called_with(
                create_model_deployment_details
            )
            mock_sync.assert_called()

    @patch.object(
        OCIDataScienceModelDeployment,
        "activate",
    )
    def test_activate(self, mock_activate):
        response = MagicMock()
        response.data = OCI_MODEL_DEPLOYMENT_RESPONSE
        mock_activate.return_value = response
        model_deployment = self.initialize_model_deployment()
        model_deployment.dsc_model_deployment.id = "test_model_deployment_id"
        model_deployment.activate(wait_for_completion=False)
        mock_activate.assert_called_with(
            wait_for_completion=False, max_wait_time=1200, poll_interval=10
        )

    @patch.object(
        OCIDataScienceModelDeployment,
        "deactivate",
    )
    def test_deactivate(self, mock_deactivate):
        response = MagicMock()
        response.data = OCI_MODEL_DEPLOYMENT_RESPONSE
        mock_deactivate.return_value = response
        model_deployment = self.initialize_model_deployment()
        model_deployment.dsc_model_deployment.id = "test_model_deployment_id"
        model_deployment.deactivate(wait_for_completion=False)
        mock_deactivate.assert_called_with(
            wait_for_completion=False, max_wait_time=1200, poll_interval=10
        )

    @patch.object(
        OCIDataScienceModelDeployment,
        "delete",
    )
    def test_delete(self, mock_delete):
        response = MagicMock()
        response.data = OCI_MODEL_DEPLOYMENT_RESPONSE
        mock_delete.return_value = response
        model_deployment = self.initialize_model_deployment()
        model_deployment.dsc_model_deployment.id = "test_model_deployment_id"
        model_deployment.delete(wait_for_completion=False)
        mock_delete.assert_called_with(
            wait_for_completion=False, max_wait_time=1200, poll_interval=10
        )

    @patch.object(OCIDataScienceMixin, "sync")
    @patch.object(
        oci.data_science.DataScienceClientCompositeOperations,
        "update_model_deployment_and_wait_for_state",
    )
    @patch.object(DataScienceModel, "create")
    def test_update(
        self,
        mock_create,
        mock_update_model_deployment_and_wait_for_state,
        mock_sync,
    ):
        dsc_model = MagicMock()
        dsc_model.id = "fakeid.datasciencemodel.oc1.iad.xxx"
        mock_create.return_value = dsc_model
        response = MagicMock()
        response.data = OCI_MODEL_DEPLOYMENT_RESPONSE
        mock_update_model_deployment_and_wait_for_state.return_value = response
        model_deployment = self.initialize_model_deployment()
        model_deployment.dsc_model_deployment.id = "test_model_deployment_id"
        update_model_deployment_details = (
            model_deployment._update_model_deployment_details()
        )
        model_deployment.update(wait_for_completion=True)
        mock_create.assert_called()
        mock_update_model_deployment_and_wait_for_state.assert_called_with(
            "test_model_deployment_id",
            update_model_deployment_details,
            wait_for_states=[
                oci.data_science.models.WorkRequest.STATUS_SUCCEEDED,
                oci.data_science.models.WorkRequest.STATUS_FAILED,
            ],
            waiter_kwargs={
                "max_interval_seconds": 10,
                "max_wait_seconds": 1200,
            },
        )
        mock_sync.assert_called()

    @patch.object(OCIDataScienceMixin, "from_ocid")
    def test_state(self, mock_from_ocid):
        mock_from_ocid.return_value = OCI_MODEL_DEPLOYMENT_RESPONSE
        model_deployment = self.initialize_model_deployment()
        model_deployment.set_spec(model_deployment.CONST_ID, "test_model_deployment_id")
        state = model_deployment.state
        mock_from_ocid.assert_called_with("test_model_deployment_id")
        assert state.name == OCI_MODEL_DEPLOYMENT_RESPONSE.lifecycle_state

    @patch.object(OCIDataScienceMixin, "from_ocid")
    def test_status(self, mock_from_ocid):
        mock_from_ocid.return_value = OCI_MODEL_DEPLOYMENT_RESPONSE
        model_deployment = self.initialize_model_deployment()
        model_deployment.set_spec(model_deployment.CONST_ID, "test_model_deployment_id")
        status = model_deployment.status
        mock_from_ocid.assert_called_with("test_model_deployment_id")
        assert status.name == OCI_MODEL_DEPLOYMENT_RESPONSE.lifecycle_state

    @patch.object(OCIDataScienceMixin, "from_ocid")
    def test_sync(self, mock_from_ocid):
        mock_from_ocid.return_value = OCI_MODEL_DEPLOYMENT_RESPONSE
        model_deployment = self.initialize_model_deployment()
        model_deployment.set_spec(model_deployment.CONST_ID, "test_model_deployment_id")
        model_deployment.sync()
        mock_from_ocid.assert_called_with("test_model_deployment_id")

    def test_random_display_name(self):
        model_deployment = self.initialize_model_deployment()
        random_name = model_deployment._random_display_name()
        assert random_name.startswith(model_deployment._PREFIX)

    @patch.object(ConsolidatedLog, "stream")
    @patch.object(OCIModelMixin, "from_ocid")
    def test_watch(self, mock_from_ocid, mock_stream):
        mock_from_ocid.return_value = OCI_MODEL_DEPLOYMENT_RESPONSE
        model_deployment = self.initialize_model_deployment()
        model_deployment.set_spec(model_deployment.CONST_ID, "test_model_deployment_id")
        model_deployment._access_log = OCILog(
            compartment_id="fakeid.compartment.oc1..xxx",
            id="fakeid.log.oc1.iad.xxx",
            log_group_id="fakeid.loggroup.oc1.iad.xxx",
            source=model_deployment.model_deployment_id,
            annotation=ModelDeploymentLogType.ACCESS,
        )
        time_start = datetime.now()
        model_deployment.watch(
            log_type="access",
            time_start=time_start,
            interval=10,
            log_filter="test_filter",
        )
        mock_from_ocid.assert_called()
        mock_stream.assert_called_with(
            source=model_deployment.model_deployment_id,
            time_start=time_start,
            stop_condition=model_deployment._stop_condition,
            interval=10,
            log_filter="test_filter",
        )

    def test_check_and_print_status(self):
        model_deployment = self.initialize_model_deployment()
        model_deployment.set_spec(model_deployment.CONST_LIFECYCLE_STATE, "ACTIVE")
        status = model_deployment._check_and_print_status("")
        assert status == "Model Deployment ACTIVE"

    def test_model_deployment_status_text(self):
        model_deployment = self.initialize_model_deployment()
        model_deployment.set_spec(model_deployment.CONST_LIFECYCLE_STATE, "INACTIVE")
        status = model_deployment._model_deployment_status_text()
        assert status == "Model Deployment INACTIVE"

    @patch.object(OCIDataScienceModelDeployment, "list")
    def test_list(self, mock_list):
        mock_list.return_value = [OCI_MODEL_DEPLOYMENT_RESPONSE]
        model_deployments = ModelDeployment.list(
            status=State.ACTIVE,
            compartment_id="test_compartment_id",
            project_id="test_project_id",
            test_arg="test",
        )
        mock_list.assert_called_with(
            status=State.ACTIVE,
            compartment_id="test_compartment_id",
            project_id="test_project_id",
            test_arg="test",
        )
        assert isinstance(model_deployments, list)
        assert isinstance(model_deployments[0], ModelDeployment)

    @patch.object(OCIDataScienceModelDeployment, "list")
    def test_list_df(self, mock_list):
        mock_list.return_value = [OCI_MODEL_DEPLOYMENT_RESPONSE]
        df = ModelDeployment.list_df(
            status=State.ACTIVE,
            compartment_id="test_compartment_id",
            project_id="test_project_id",
        )
        mock_list.assert_called_with(
            status=State.ACTIVE,
            compartment_id="test_compartment_id",
            project_id="test_project_id",
        )
        assert isinstance(df, pandas.DataFrame)

    def test_model_deployment_with_subnet_id(self):
        model_deployment = self.initialize_model_deployment()
        assert model_deployment.infrastructure.subnet_id == None

        model_deployment.infrastructure.with_subnet_id("test_id")
        assert model_deployment.infrastructure.subnet_id == "test_id"

    def test_update_spec(self):
        model_deployment = self.initialize_model_deployment()
        model_deployment._update_spec(
            display_name="test_updated_name",
            freeform_tags={"test_updated_key":"test_updated_value"},
            access_log={
                "log_id": "test_updated_access_log_id"
            },
            predict_log={
                "log_group_id": "test_updated_predict_log_group_id"
            },
            shape_config_details={
                "ocpus": 100,
                "memoryInGBs": 200
            },
            replica=20,
            image="test_updated_image",
            env={
                "test_updated_env_key":"test_updated_env_value"
            }
        )

        assert model_deployment.display_name == "test_updated_name"
        assert model_deployment.freeform_tags == {
            "test_updated_key":"test_updated_value"
        }
        assert model_deployment.infrastructure.access_log == {
            "logId": "test_updated_access_log_id",
            "logGroupId": "fakeid.loggroup.oc1.iad.xxx"
        }
        assert model_deployment.infrastructure.predict_log == {
            "logId": "fakeid.log.oc1.iad.xxx",
            "logGroupId": "test_updated_predict_log_group_id"
        }
        assert model_deployment.infrastructure.shape_config_details == {
            "ocpus": 100,
            "memoryInGBs": 200
        }
        assert model_deployment.infrastructure.replica == 20
        assert model_deployment.runtime.image == "test_updated_image"
        assert model_deployment.runtime.env == {
            "test_updated_env_key":"test_updated_env_value"
        }

    @patch.object(OCIDataScienceMixin, "sync")
    @patch.object(
        oci.data_science.DataScienceClient,
        "create_model_deployment",
    )
    @patch.object(DataScienceModel, "create")
    def test_model_deployment_with_large_size_artifact(
        self, 
        mock_create, 
        mock_create_model_deployment, 
        mock_sync
    ):
        dsc_model = MagicMock()
        dsc_model.id = "fakeid.datasciencemodel.oc1.iad.xxx"
        mock_create.return_value = dsc_model
        model_deployment = self.initialize_model_deployment()
        (
            model_deployment.runtime
            .with_auth({"test_key":"test_value"})
            .with_region("test_region")
            .with_overwrite_existing_artifact(True)
            .with_remove_existing_artifact(True)
            .with_timeout(100)
            .with_bucket_uri("test_bucket_uri")
        )

        runtime_dict = model_deployment.runtime.to_dict()["spec"]
        assert runtime_dict["auth"] == {"test_key": "test_value"}
        assert runtime_dict["region"] == "test_region"
        assert runtime_dict["overwriteExistingArtifact"] == True
        assert runtime_dict["removeExistingArtifact"] == True
        assert runtime_dict["timeout"] == 100
        assert runtime_dict["bucketUri"] == "test_bucket_uri"

        response = MagicMock()
        response.data = OCI_MODEL_DEPLOYMENT_RESPONSE
        mock_create_model_deployment.return_value = response
        model_deployment = self.initialize_model_deployment()
        model_deployment.set_spec(model_deployment.CONST_ID, "test_model_deployment_id")
        
        create_model_deployment_details = (
            model_deployment._build_model_deployment_details()
        )
        model_deployment.deploy(wait_for_completion=False)
        mock_create.assert_called_with(
            bucket_uri="test_bucket_uri",
            auth={"test_key":"test_value"},
            region="test_region",
            overwrite_existing_artifact=True,
            remove_existing_artifact=True,
            timeout=100
        )
        mock_create_model_deployment.assert_called_with(create_model_deployment_details)
        mock_sync.assert_called()
