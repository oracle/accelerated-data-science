#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import oci
import pytest
from unittest.mock import MagicMock, patch
from oci.data_science.models import (
    ModelDeployment,
)
from ads.common.oci_datascience import OCIDataScienceMixin
from ads.common.oci_mixin import OCIModelMixin, OCIWorkRequestMixin

from ads.model.service.oci_datascience_model_deployment import (
    ACTIVATE_WORKFLOW_STEPS,
    CREATE_WORKFLOW_STEPS,
    DEACTIVATE_WORKFLOW_STEPS,
    DELETE_WORKFLOW_STEPS,
    OCIDataScienceModelDeployment,
)

MODEL_DEPLOYMENT_OCID = "fakeid.datasciencemodeldeployment.oc1.iad.xxx"

OCI_MODEL_DEPLOYMENT_PAYLOAD = {
    "id": MODEL_DEPLOYMENT_OCID,
    "compartment_id": "fakeid.compartment.oc1..xxx",
    "project_id": "fakeid.datascienceproject.oc1.iad.xxx",
    "display_name": "Generic Model Deployment With Small Artifact new",
    "description": "The model deployment description",
    "lifecycle_state": "ACTIVE",
    "lifecycle_details": "Model Deployment is Active.",
    "created_by": "fakeid.user.oc1..xxx",
    "freeform_tags": {"key1": "value1"},
    "defined_tags": {"key1": {"skey1": "value1"}},
    "time_created": "2022-08-24T17:07:39.200000Z",
    "model_deployment_configuration_details": {
        "deployment_type": "SINGLE_MODEL",
        "model_configuration_details": {
            "model_id": "fakeid.datasciencemodel.oc1.iad.xxx",
            "instance_configuration": {
                "instance_shape_name": "VM.Standard.E4.Flex",
                "model_deployment_instance_shape_config_details": {
                    "ocpus": 10,
                    "memory_in_gbs": 36,
                },
            },
            "scaling_policy": {
                "policy_type": "FIXED_SIZE",
                "instance_count": 5,
            },
            "bandwidth_mbps": 5,
        },
        "stream_configuration_details": {
            "input_stream_ids": ["123", "456"],
            "output_stream_ids": ["321", "654"],
        },
        "environment_configuration_details": {
            "environment_configuration_type": "DEFAULT",
            "environment_variables": {
                "key": "value",
            },
        },
    },
    "category_log_details": {
        "access": {
            "log_id": "fakeid.log.oc1.iad.xxx",
            "log_group_id": "fakeid.loggroup.oc1.iad.xxx",
        },
        "predict": {
            "log_id": "fakeid.log.oc1.iad.xxx",
            "log_group_id": "fakeid.loggroup.oc1.iad.xxx",
        },
    },
    "model_deployment_url": "model_deployment_url",
}


class TestOCIDataScienceModelDeployment:
    def setup_method(self):
        self.mock_model_deployment = OCIDataScienceModelDeployment(
            **OCI_MODEL_DEPLOYMENT_PAYLOAD
        )

    def test_activate(self):
        with patch.object(OCIDataScienceModelDeployment, "from_id") as mock_from_id:
            response = copy.deepcopy(OCI_MODEL_DEPLOYMENT_PAYLOAD)
            mock_from_id.return_value = OCIDataScienceModelDeployment(
                **response
            )
            with pytest.raises(
                Exception,
                match=f"Model deployment {self.mock_model_deployment.id} is already in active state."
            ):
                self.mock_model_deployment.activate(
                    wait_for_completion=False,
                    max_wait_time=1,
                    poll_interval=1,
                )

            response["lifecycle_state"] = "FAILED"
            mock_from_id.return_value = OCIDataScienceModelDeployment(
                **response
            )
            with pytest.raises(
                Exception,
                match=f"Can't activate model deployment {self.mock_model_deployment.id} when it's in FAILED state."
            ):
                self.mock_model_deployment.activate(
                    wait_for_completion=False,
                    max_wait_time=1,
                    poll_interval=1,
                )

            response["lifecycle_state"] = "INACTIVE"
            mock_from_id.return_value = OCIDataScienceModelDeployment(
                **response
            )
            with patch.object(
                oci.data_science.DataScienceClient,
                "activate_model_deployment",
            ) as mock_activate:
                with patch.object(OCIDataScienceModelDeployment, "sync") as mock_sync:
                    self.mock_model_deployment.activate(
                        wait_for_completion=False,
                        max_wait_time=1,
                        poll_interval=1,
                    )

                    mock_activate.assert_called_with(self.mock_model_deployment.id)
                    mock_sync.assert_called()
                    mock_from_id.assert_called_with(self.mock_model_deployment.id)

    def test_activate_with_waiting(self):
        with patch.object(OCIDataScienceModelDeployment, "from_id") as mock_from_id:
            response = copy.deepcopy(OCI_MODEL_DEPLOYMENT_PAYLOAD)
            response["lifecycle_state"] = "INACTIVE"
            mock_from_id.return_value = OCIDataScienceModelDeployment(
                **response
            )
            with patch.object(
                oci.data_science.DataScienceClient,
                "activate_model_deployment",
            ) as mock_activate:
                response = MagicMock()
                response.headers = {
                    "opc-work-request-id": "test",
                }
                mock_activate.return_value = response
                with patch.object(
                    OCIWorkRequestMixin, "wait_for_progress"
                ) as mock_wait:
                    with patch.object(
                        OCIDataScienceModelDeployment, "sync"
                    ) as mock_sync:
                        self.mock_model_deployment.activate(
                            max_wait_time=1,
                            poll_interval=1,
                        )

                    mock_activate.assert_called_with(self.mock_model_deployment.id)
                    mock_wait.assert_called_with(
                        "test",
                        ACTIVATE_WORKFLOW_STEPS,
                        1,
                        1,
                    )
                    mock_sync.assert_called()

    def test_deactivate(self):
        with patch.object(OCIDataScienceModelDeployment, "from_id") as mock_from_id:
            response = copy.deepcopy(OCI_MODEL_DEPLOYMENT_PAYLOAD)
            response["lifecycle_state"] = "INACTIVE"
            mock_from_id.return_value = OCIDataScienceModelDeployment(
                **response
            )
            with pytest.raises(
                Exception,
                match=f"Model deployment {self.mock_model_deployment.id} is already in inactive state."
            ):
                self.mock_model_deployment.deactivate(
                    wait_for_completion=False,
                    max_wait_time=1,
                    poll_interval=1,
                )

            response["lifecycle_state"] = "FAILED"
            mock_from_id.return_value = OCIDataScienceModelDeployment(
                **response
            )
            with pytest.raises(
                Exception,
                match=f"Can't deactivate model deployment {self.mock_model_deployment.id} when it's in FAILED state."
            ):
                self.mock_model_deployment.deactivate(
                    wait_for_completion=False,
                    max_wait_time=1,
                    poll_interval=1,
                )
            response["lifecycle_state"] = "ACTIVE"
            mock_from_id.return_value = OCIDataScienceModelDeployment(
                **response
            )
            with patch.object(
                oci.data_science.DataScienceClient,
                "deactivate_model_deployment",
            ) as mock_deactivate:
                with patch.object(OCIDataScienceModelDeployment, "sync") as mock_sync:
                    self.mock_model_deployment.deactivate(
                        wait_for_completion=False,
                        max_wait_time=1,
                        poll_interval=1,
                    )

                    mock_deactivate.assert_called_with(self.mock_model_deployment.id)
                    mock_sync.assert_called()
                    mock_from_id.assert_called_with(self.mock_model_deployment.id)

    def test_deactivate_with_waiting(self):
        with patch.object(OCIDataScienceModelDeployment, "from_id") as mock_from_id:
            mock_from_id.return_value = OCIDataScienceModelDeployment(
                **OCI_MODEL_DEPLOYMENT_PAYLOAD
            )
            with patch.object(
                oci.data_science.DataScienceClient,
                "deactivate_model_deployment",
            ) as mock_deactivate:
                response = MagicMock()
                response.headers = {
                    "opc-work-request-id": "test",
                }
                mock_deactivate.return_value = response
                with patch.object(
                    OCIWorkRequestMixin, "wait_for_progress"
                ) as mock_wait:
                    with patch.object(
                        OCIDataScienceModelDeployment, "sync"
                    ) as mock_sync:
                        self.mock_model_deployment.deactivate(
                            max_wait_time=1,
                            poll_interval=1,
                        )

                    mock_deactivate.assert_called_with(
                        self.mock_model_deployment.id
                    )
                    mock_wait.assert_called_with(
                        "test",
                        DEACTIVATE_WORKFLOW_STEPS,
                        1,
                        1,
                    )
                    mock_sync.assert_called()

    def test_create(self):
        with patch.object(
            oci.data_science.DataScienceClient,
            "create_model_deployment",
        ) as mock_create:
            with patch.object(
                OCIDataScienceModelDeployment, "to_oci_model"
            ) as mock_to_oci_mode:
                with patch.object(
                    OCIDataScienceMixin, "update_from_oci_model"
                ) as mock_update_from_oci_model:
                    create_model_deployment_details = MagicMock()
                    oci_model_deployment = ModelDeployment(
                        **OCI_MODEL_DEPLOYMENT_PAYLOAD
                    )
                    mock_to_oci_mode.return_value = oci_model_deployment
                    with patch.object(
                        OCIDataScienceModelDeployment, "sync"
                    ) as mock_sync:
                        self.mock_model_deployment.create(
                            create_model_deployment_details,
                            wait_for_completion=False,
                            max_wait_time=1,
                            poll_interval=1,
                        )

                        mock_create.assert_called_with(
                            create_model_deployment_details,
                        )
                        mock_update_from_oci_model.assert_called()
                        mock_sync.assert_called()

    def test_create_with_waiting(self):
        with patch.object(
            oci.data_science.DataScienceClient,
            "create_model_deployment",
        ) as mock_create:
            response = MagicMock()
            response.headers = {
                "opc-work-request-id": "test",
            }
            mock_create.return_value = response
            with patch.object(
                OCIDataScienceModelDeployment, "to_oci_model"
            ) as mock_to_oci_mode:
                with patch.object(
                    OCIDataScienceMixin, "update_from_oci_model"
                ) as mock_update_from_oci_model:
                    oci_model_deployment = ModelDeployment(
                        **OCI_MODEL_DEPLOYMENT_PAYLOAD
                    )
                    mock_to_oci_mode.return_value = oci_model_deployment
                    with patch.object(
                        OCIWorkRequestMixin, "wait_for_progress"
                    ) as mock_wait:
                        with patch("json.loads") as mock_json_load:
                            create_model_deployment_details = MagicMock()
                            mock_json_load.return_value = {
                                "lifecycle_state": "UNKNOWN",
                                "id": self.mock_model_deployment.id,
                            }
                            with patch.object(
                                OCIDataScienceModelDeployment, "sync"
                            ) as mock_sync:
                                self.mock_model_deployment.create(
                                    create_model_deployment_details,
                                    max_wait_time=1,
                                    poll_interval=1,
                                )

                                mock_create.assert_called_with(
                                    create_model_deployment_details,
                                )
                                mock_update_from_oci_model.assert_called()
                                mock_wait.assert_called_with(
                                    "test",
                                    CREATE_WORKFLOW_STEPS,                                    
                                    1,
                                    1,
                                )
                                mock_sync.assert_called()

    def test_update(self):
        with patch.object(
            oci.data_science.DataScienceClientCompositeOperations,
            "update_model_deployment_and_wait_for_state",
        ) as mock_update:
            with patch.object(OCIDataScienceModelDeployment, "sync") as mock_sync:
                update_model_deployment_details = MagicMock()
                update_model_deployment_details.display_name = "test_name"
                update_model_deployment_details.description = "test_description"
                self.mock_model_deployment.update(
                    update_model_deployment_details=update_model_deployment_details,
                    wait_for_completion=False,
                    max_wait_time=1,
                    poll_interval=1,
                )

                mock_update.assert_called_with(
                    self.mock_model_deployment.id,
                    update_model_deployment_details,
                    wait_for_states=[],
                    waiter_kwargs={
                        "max_interval_seconds": 1,
                        "max_wait_seconds": 1,
                    },
                )
                mock_sync.assert_called()

    def test_update_with_waiting(self):
        with patch.object(
            oci.data_science.DataScienceClientCompositeOperations,
            "update_model_deployment_and_wait_for_state",
        ) as mock_update:
            with patch.object(OCIDataScienceModelDeployment, "sync") as mock_sync:
                update_model_deployment_details = MagicMock()
                update_model_deployment_details.display_name = "test_name"
                update_model_deployment_details.description = "test_description"
                self.mock_model_deployment.update(
                    update_model_deployment_details=update_model_deployment_details,
                    max_wait_time=1,
                    poll_interval=1,
                )

                mock_update.assert_called_with(
                    self.mock_model_deployment.id,
                    update_model_deployment_details,
                    wait_for_states=[
                        oci.data_science.models.WorkRequest.STATUS_SUCCEEDED,
                        oci.data_science.models.WorkRequest.STATUS_FAILED,
                    ],
                    waiter_kwargs={
                        "max_interval_seconds": 1,
                        "max_wait_seconds": 1,
                    },
                )
                mock_sync.assert_called()

    def test_delete(self):
        with patch.object(OCIDataScienceModelDeployment, "from_id") as mock_from_id:
            response = copy.deepcopy(OCI_MODEL_DEPLOYMENT_PAYLOAD)
            response["lifecycle_state"] = "DELETED"
            mock_from_id.return_value = OCIDataScienceModelDeployment(
                **response
            )
            with pytest.raises(
                Exception,
                match=f"Model deployment {self.mock_model_deployment.id} is either deleted or being deleted."
            ):
                self.mock_model_deployment.delete(
                    wait_for_completion=False,
                    max_wait_time=1,
                    poll_interval=1,
                )

            response["lifecycle_state"] = "UPDATING"
            mock_from_id.return_value = OCIDataScienceModelDeployment(
                **response
            )
            with pytest.raises(
                Exception,
                match=f"Can't delete model deployment {self.mock_model_deployment.id} when it's in UPDATING state."
            ):
                self.mock_model_deployment.delete(
                    wait_for_completion=False,
                    max_wait_time=1,
                    poll_interval=1,
                )
            response["lifecycle_state"] = "ACTIVE"
            mock_from_id.return_value = OCIDataScienceModelDeployment(
                **response
            )
            with patch.object(
                oci.data_science.DataScienceClient,
                "delete_model_deployment",
            ) as mock_delete:
                with patch.object(OCIDataScienceModelDeployment, "sync") as mock_sync:
                    self.mock_model_deployment.delete(
                        wait_for_completion=False,
                        max_wait_time=1,
                        poll_interval=1,
                    )

                    mock_delete.assert_called_with(self.mock_model_deployment.id)
                    mock_sync.assert_called()
                    mock_from_id.assert_called_with(self.mock_model_deployment.id)

    def test_delete_with_waiting(self):
        with patch.object(OCIDataScienceModelDeployment, "from_id") as mock_from_id:
            mock_from_id.return_value = OCIDataScienceModelDeployment(
                **OCI_MODEL_DEPLOYMENT_PAYLOAD
            )
            with patch.object(
                OCIWorkRequestMixin, "wait_for_progress"
            ) as mock_wait:
                with patch.object(
                    oci.data_science.DataScienceClient,
                    "delete_model_deployment",
                ) as mock_delete:
                    response = MagicMock()
                    response.headers = {
                        "opc-work-request-id": "test",
                    }
                    mock_delete.return_value = response
                    with patch.object(
                        OCIDataScienceModelDeployment, "sync"
                    ) as mock_sync:
                        self.mock_model_deployment.delete(
                            max_wait_time=1, poll_interval=1
                        )

                        mock_delete.assert_called_with(self.mock_model_deployment.id)
                        mock_wait.assert_called_with(
                            "test",
                            DELETE_WORKFLOW_STEPS,
                            1,
                            1,
                        )
                        mock_sync.assert_called()

    @patch.object(OCIModelMixin, "from_ocid")
    def test_from_id(self, mock_from_ocid):
        """Tests getting a model by OCID."""
        OCIDataScienceModelDeployment.from_id(self.mock_model_deployment.id)
        mock_from_ocid.assert_called_with(self.mock_model_deployment.id)

    @patch.object(oci.pagination, "list_call_get_all_results")
    def test_list(self, mock_list_call_get_all_results):
        response = MagicMock()
        response.data = [MagicMock()]
        mock_list_call_get_all_results.return_value = response
        model_deployments = OCIDataScienceModelDeployment.list(
            status="ACTIVE",
            compartment_id="test_compartment_id",
            project_id="test_project_id",
            test_arg="test",
        )
        mock_list_call_get_all_results.assert_called()
        assert isinstance(model_deployments, list)
