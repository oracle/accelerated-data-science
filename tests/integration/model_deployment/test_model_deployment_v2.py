#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import unittest

from ads.model.deployment import ModelDeployment
from ads.model.deployment.model_deployment_infrastructure import (
    ModelDeploymentInfrastructure,
)
from ads.model.deployment.model_deployment_runtime import (
    ModelDeploymentContainerRuntime,
)
from ads.model.service.oci_datascience_model_deployment import (
    OCIDataScienceModelDeployment,
)
from tests.integration.config import secrets

try:
    from oci.data_science.models import (
        OcirModelDeploymentEnvironmentConfigurationDetails,
        UpdateOcirModelDeploymentEnvironmentConfigurationDetails,
    )
except (ImportError, AttributeError) as e:
    raise unittest.SkipTest(
        "Support for OCI ModelDeployment BYOC is not available. Skipping the ModelDeployment tests."
    )

COMPARTMENT_ID = secrets.common.COMPARTMENT_ID
PROJECT_ID = secrets.common.PROJECT_OCID
MODEL_URI = secrets.model_deployment.MODEL_OCID_MD
CONTAINER_IMAGE = secrets.model_deployment.MODEL_DEPLOYMENT_BYOC_IMAGE
CONTAINER_IMAGE_DIGEST = secrets.model_deployment.CONTAINER_IMAGE_DIGEST
ENTRYPOINT = ["python", "/opt/ds/model/deployed_model/api.py"]
SERVER_PORT = 5000
HEALTH_CHECK_PORT = 5000

LOG_GROUP_ID = secrets.common.LOG_GROUP_ID
LOG_ID = secrets.model_deployment.LOG_OCID

MODEL_DEPLOYMENT_DISPLAY_NAME_CODE = "test_model_deployment_code"
UPDATED_MODEL_DEPLOYMENT_DISPLAY_NAME_CODE = "updated_test_model_deployment_code"


class ModelDeploymentBYOCTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        deployments = ModelDeployment.list(
            compartment_id=COMPARTMENT_ID, project_id=PROJECT_ID
        )
        for deployment in deployments:
            if deployment.lifecycle_state not in [
                "DELETING",
                "DELETED",
                "CREATING",
                "UPDATING",
            ] and deployment.display_name in [
                MODEL_DEPLOYMENT_DISPLAY_NAME_CODE,
                UPDATED_MODEL_DEPLOYMENT_DISPLAY_NAME_CODE,
            ]:
                deployment.delete(wait_for_completion=False)
        return super().tearDownClass()

    def initialize_model_deployment_from_code(self):
        infrastructure = (
            ModelDeploymentInfrastructure()
            .with_project_id(PROJECT_ID)
            .with_compartment_id(COMPARTMENT_ID)
            .with_shape_name("VM.Standard.E4.Flex")
            .with_shape_config_details(ocpus=1, memory_in_gbs=16)
            .with_replica(1)
            .with_bandwidth_mbps(10)
            .with_web_concurrency(10)
            .with_access_log(log_group_id=LOG_GROUP_ID, log_id=LOG_ID)
            .with_predict_log(log_group_id=LOG_GROUP_ID, log_id=LOG_ID)
        )

        container_runtime = (
            ModelDeploymentContainerRuntime()
            .with_image(CONTAINER_IMAGE)
            .with_image_digest(CONTAINER_IMAGE_DIGEST)
            .with_entrypoint(ENTRYPOINT)
            .with_server_port(SERVER_PORT)
            .with_health_check_port(HEALTH_CHECK_PORT)
            .with_env({"key": "value"})
            .with_deployment_mode("HTTPS_ONLY")
            .with_model_uri(MODEL_URI)
        )

        model_deployment = (
            ModelDeployment()
            .with_display_name(MODEL_DEPLOYMENT_DISPLAY_NAME_CODE)
            .with_description("The model deployment v2 integration from code")
            .with_freeform_tags(key1="value1")
            .with_infrastructure(infrastructure)
            .with_runtime(container_runtime)
        )

        return model_deployment

    def test_predict_deploy_update_deactivate_activate_delete_from_code(self):
        model_deployment = self.initialize_model_deployment_from_code()
        try:
            model_deployment.deploy(wait_for_completion=True)

            assert model_deployment.model_deployment_id != None
            assert model_deployment.time_created != None
            assert model_deployment.lifecycle_details != None
            assert (
                model_deployment.lifecycle_state
                == OCIDataScienceModelDeployment.LIFECYCLE_STATE_ACTIVE
            )
            assert model_deployment.url != None

            result = model_deployment.predict(data={"line": "12"})
            assert result == {"prediction LDA": 21}

            assert model_deployment.display_name == MODEL_DEPLOYMENT_DISPLAY_NAME_CODE

            model_deployment.with_display_name(
                UPDATED_MODEL_DEPLOYMENT_DISPLAY_NAME_CODE
            )
            model_deployment.update(wait_for_completion=True)

            assert (
                model_deployment.display_name
                == UPDATED_MODEL_DEPLOYMENT_DISPLAY_NAME_CODE
            )

            model_deployment.deactivate(wait_for_completion=True)
            assert (
                model_deployment.lifecycle_state
                == OCIDataScienceModelDeployment.LIFECYCLE_STATE_INACTIVE
            )

            model_deployment.activate(wait_for_completion=True)
            assert (
                model_deployment.lifecycle_state
                == OCIDataScienceModelDeployment.LIFECYCLE_STATE_ACTIVE
            )

            model_deployment.delete(wait_for_completion=True)
            assert (
                model_deployment.lifecycle_state
                == OCIDataScienceModelDeployment.LIFECYCLE_STATE_DELETED
            )
        except Exception as ex:
            print("Model deploy failed with error: %s", ex)
            exit(1)
