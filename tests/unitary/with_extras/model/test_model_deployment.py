#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest.mock import MagicMock, patch

import oci
import pytest
from ads.common.oci_logging import ConsolidatedLog, OCILog
from ads.common.oci_mixin import OCIModelMixin
from ads.model.deployment.model_deployment import (
    LogNotConfiguredError,
    ModelDeployment,
    ModelDeploymentLogType,
)

OCI_LOG_DETAILS = {
    "id": "ocid1.log.oc1.iad.<unique_ocid>",
    "log_group_id": "ocid1.loggroup.oc1.iad.<unique_ocid>",
    "compartment_id": "ocid1.compartment.oc1..<unique_ocid>",
}


class TestModelDeployment:
    def setup_method(self):
        self.test_model_deployment = ModelDeployment(
            model_deployment_id="test_model_deployment_id", properties={"key": "value"}
        )
        self.test_model_deployment._access_log = None

    @patch.object(ModelDeployment, "_log_details")
    def test_logs_access(self, mock_log_details):
        """Tests getting the access or predict logs."""
        with patch.object(OCIModelMixin, "deserialize") as mock_deserialize:
            mock_deserialize.return_value = oci.logging.models.Log(**OCI_LOG_DETAILS)

            log_details_obj = MagicMock()
            log_details_obj.log_id = OCI_LOG_DETAILS["id"]
            log_details_obj.log_group_id = OCI_LOG_DETAILS["log_group_id"]

            mock_log_details.return_value = log_details_obj

            test_result = self.test_model_deployment.logs(
                log_type=ModelDeploymentLogType.ACCESS
            )
            mock_log_details.assert_called_with(log_type=ModelDeploymentLogType.ACCESS)
            access_log = test_result.logging_instance[0]
            assert isinstance(test_result, ConsolidatedLog)
            assert isinstance(access_log, OCILog)
            assert access_log.id == OCI_LOG_DETAILS["id"]
            assert access_log.log_group_id == OCI_LOG_DETAILS["log_group_id"]

    @patch.object(ModelDeployment, "_log_details")
    def test_logs_predict(self, mock_log_details):
        with patch.object(OCIModelMixin, "deserialize") as mock_deserialize:

            mock_deserialize.return_value = oci.logging.models.Log(**OCI_LOG_DETAILS)
            log_details_obj = MagicMock()
            log_details_obj.log_id = OCI_LOG_DETAILS["id"]
            log_details_obj.log_group_id = OCI_LOG_DETAILS["log_group_id"]

            mock_log_details.return_value = log_details_obj
            test_result = self.test_model_deployment.logs(
                log_type=ModelDeploymentLogType.PREDICT
            )
            mock_log_details.assert_called_with(log_type=ModelDeploymentLogType.PREDICT)
            predict_log = test_result.logging_instance[0]
            assert isinstance(test_result, ConsolidatedLog)
            assert isinstance(predict_log, OCILog)
            assert predict_log.id == OCI_LOG_DETAILS["id"]
            assert predict_log.log_group_id == OCI_LOG_DETAILS["log_group_id"]

    def test_logs_fail(self):
        with pytest.raises(LogNotConfiguredError):
            self.test_model_deployment.logs(log_type=ModelDeploymentLogType.ACCESS)

        with pytest.raises(LogNotConfiguredError):
            self.test_model_deployment.logs(log_type=ModelDeploymentLogType.PREDICT)

        with pytest.raises(
            LogNotConfiguredError,
            match="Neither `predict` nor `access` log was configured for the model deployment.",
        ):
            self.test_model_deployment.logs()

        with pytest.raises(
            ValueError,
            match="Parameter log_type should be either access, predict or None.",
        ):
            self.test_model_deployment.logs(log_type="unrecognized_log_type")

    @patch.object(ModelDeployment, "_log_details")
    def test_predict_log(self, mock_log_details):
        """Tests getting the model deployment predict logs object."""
        log_details_obj = MagicMock()
        log_details_obj.log_id = OCI_LOG_DETAILS["log_group_id"]
        log_details_obj.log_group_id = OCI_LOG_DETAILS["log_group_id"]

        mock_log_details.return_value = log_details_obj

        test_log = self.test_model_deployment.predict_log
        mock_log_details.assert_called_with(log_type=ModelDeploymentLogType.PREDICT)
        assert isinstance(test_log, OCILog)
        test_log.source == "test_model_deployment_id"
        test_log.id = OCI_LOG_DETAILS["log_group_id"]
        test_log.log_group_id = OCI_LOG_DETAILS["log_group_id"]

    @patch.object(ModelDeployment, "_log_details")
    def test_access_log(self, mock_log_details):
        """Tests getting the model deployment access logs object."""
        log_details_obj = MagicMock()
        log_details_obj.log_id = OCI_LOG_DETAILS["log_group_id"]
        log_details_obj.log_group_id = OCI_LOG_DETAILS["log_group_id"]

        mock_log_details.return_value = log_details_obj

        test_log = self.test_model_deployment.access_log
        mock_log_details.assert_called_with(log_type=ModelDeploymentLogType.ACCESS)
        assert isinstance(test_log, OCILog)
        test_log.source == "test_model_deployment_id"
        test_log.id = OCI_LOG_DETAILS["log_group_id"]
        test_log.log_group_id = OCI_LOG_DETAILS["log_group_id"]

    def test__log_details(self):
        """Tests getting log details for the provided `log_type`."""

        with pytest.raises(LogNotConfiguredError):
            self.test_model_deployment._log_details(
                log_type=ModelDeploymentLogType.ACCESS
            )

        mock_category_log_details = MagicMock()
        expected_result = MagicMock()
        mock_category_log_details.access = expected_result
        self.test_model_deployment.properties.category_log_details = (
            mock_category_log_details
        )

        test_result = self.test_model_deployment._log_details(
            log_type=ModelDeploymentLogType.ACCESS
        )

        assert test_result == expected_result
