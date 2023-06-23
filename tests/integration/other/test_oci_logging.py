#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Contains integration test for OCI logging integration on ociodscdev tenancy.
"""
import datetime
import importlib
import io
import logging
import os
import pytest
import sys
import time
import unittest
import uuid

import oci

from ads import config
from ads.common.oci_logging import ConsolidatedLog, OCILog, OCILogGroup
from ads.model.deployment import ModelDeployer
from ads.model.deployment.model_deployment import ModelDeploymentLogType
from tests.integration.config import secrets

# logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


class StopCounter:
    counter = 0

    def check(self):
        self.counter += 1
        if self.counter > 5:
            return True
        return False


class OCILoggingTestCase(unittest.TestCase):
    # Compartment ID for the networks compartment
    COMPARTMENT_ID = secrets.common.COMPARTMENT_ID
    LOG_ID = secrets.jobs.LOG_ID
    existing_env = {}
    LOG_GROUP_NAME_PREFIX = "ADS-int-test-loggroup"

    @classmethod
    def setUpClass(cls) -> None:
        logger.debug("Setting up logging test...")
        # Save existing env vars
        cls.existing_env = os.environ.copy()
        # Generate names
        cls.log_group_name = f"{cls.LOG_GROUP_NAME_PREFIX}-{uuid.uuid4()}"
        # Set compartment ID
        os.environ["NB_SESSION_COMPARTMENT_OCID"] = cls.COMPARTMENT_ID
        importlib.reload(config)
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        logger.debug("Cleaning up resource...")
        # Clean up resource
        client = oci.logging.LoggingManagementClient(oci.config.from_file())
        log_groups = oci.pagination.list_call_get_all_results(
            client.list_log_groups,
            cls.COMPARTMENT_ID,
        ).data

        # Log groups created 12+ hours ago are assumed to be left by other tests.
        # Normally this would not happen but we will remove them if they exist.
        time_to_delete = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(hours=12)

        for log_group in log_groups:
            if not str(log_group.display_name).startswith(
                cls.LOG_GROUP_NAME_PREFIX
            ) or (
                # Log groups might be created by other tests running in parallel,
                # We only delete the one created by this test.
                log_group.time_created > time_to_delete
                and log_group.display_name != cls.log_group_name
            ):
                continue

            logs = client.list_logs(log_group.id).data
            for log in logs:
                client.delete_log(log_group.id, log.id)
            time.sleep(3)
            client.delete_log_group(log_group.id)

        # Restore env vars
        if "NB_SESSION_COMPARTMENT_OCID" in cls.existing_env:
            os.environ["NB_SESSION_COMPARTMENT_OCID"] = cls.existing_env[
                "NB_SESSION_COMPARTMENT_OCID"
            ]
        else:
            os.environ.pop("NB_SESSION_COMPARTMENT_OCID", None)
        importlib.reload(config)
        return super().tearDownClass()

    def test_oci_logging(self):
        # Create log group
        log_group = OCILogGroup(display_name=self.log_group_name)
        # id should be None before it is created
        self.assertIsNone(log_group.id)
        # Create the log group with OCI
        log_group.create()
        # Compartment ID should be loaded from env var after create() is called
        self.assertEqual(log_group.compartment_id, self.COMPARTMENT_ID)
        # id should not be None once created
        self.assertIsNotNone(log_group.id)
        # Create log
        log_name_1 = f"ADS-int-test-log-{uuid.uuid4()}"
        log_name_2 = f"ADS-int-test-log-{uuid.uuid4()}"
        try:
            log_1 = log_group.create_log(log_name_1)
            log_2 = log_group.create_log(log_name_2)
        except oci.exceptions.ServiceError as ex:
            # Skip the test is resource limit is reached.
            if ex.code == "LimitExceeded":
                self.skipTest(ex.message)
            raise ex
        self.assertIsNotNone(log_1.id)
        self.assertIsNotNone(log_2.id)
        # Get logs with log group
        logs = log_group.list_logs()
        self.assertIsInstance(logs, list)
        self.assertEqual(len(logs), 2)
        # Check log names
        log_names = [log.display_name for log in logs]
        self.assertIn(log_name_1, log_names)
        self.assertIn(log_name_2, log_names)
        # Test get log group with OCID
        log_group = OCILogGroup.from_ocid(log_group.id)
        self.assertEqual(log_group.display_name, self.log_group_name)
        # Test get log group by name
        log_group = OCILogGroup.from_name(self.log_group_name)
        self.assertIsNotNone(log_group)
        self.assertEqual(log_group.display_name, self.log_group_name)
        # Delete the log group and logs
        log_group.delete()

    def test_invalid_log_tail(self):
        oci_log = OCILog.from_ocid(self.LOG_ID)
        tail = oci_log.tail(
            source="wrong_source_ocid",
            limit=1,
        )
        assert len(tail) == 0

    def test_multiple_log_tail(self):
        oci_log = OCILog.from_ocid(self.LOG_ID)
        tail_length = 10
        tail = oci_log.tail(
            limit=tail_length,
        )
        assert len(tail) == tail_length

    def test_log_streaming(self):
        oci_log = OCILog.from_ocid(secrets.jobs.LOG_ID)

        oci_log.stream(
            interval=1,
            stop_condition=StopCounter().check,
        )

    def test_log_obj_serialization(self):
        oci_log = OCILog.from_ocid(self.LOG_ID)
        data = oci_log.to_dict()
        self.assertIsInstance(data, dict)
        self.assertIn("id", data)
        self.assertIn("displayName", data)
        self.assertEqual(oci_log.name, data.get("displayName"))
        oci_log = OCILog.from_dict(data)
        self.assertIsNotNone(oci_log.display_name)
        self.assertEqual(len(data), len(oci_log.to_dict(flatten=True)))


class ConsolidatedLoggingTestCase(unittest.TestCase):
    COMPARTMENT_ID = secrets.common.COMPARTMENT_ID
    LOG_GROUP_ID = secrets.other.LOG_GROUP_ID
    ACCESS_LOG_ID = secrets.other.ACCESS_LOG_ID
    PREDICT_LOG_ID = secrets.other.PREDICT_LOG_ID
    MODEL_DEPLOYMENT_ID = secrets.other.MODEL_DEPLOYMENT_ID

    @classmethod
    def setUpClass(cls) -> None:
        model_deployment = ModelDeployer().get_model_deployment(cls.MODEL_DEPLOYMENT_ID)
        model_deployment.predict([1, 2, 3])
        return super().setUpClass()

    def construct_consolidated_log(self):
        access_log = OCILog(
            compartment_id=self.COMPARTMENT_ID,
            id=self.ACCESS_LOG_ID,
            log_group_id=self.LOG_GROUP_ID,
            source=self.MODEL_DEPLOYMENT_ID,
            annotation=ModelDeploymentLogType.ACCESS,
        )

        predict_log = OCILog(
            compartment_id=self.COMPARTMENT_ID,
            id=self.PREDICT_LOG_ID,
            log_group_id=self.LOG_GROUP_ID,
            source=self.MODEL_DEPLOYMENT_ID,
            annotation=ModelDeploymentLogType.PREDICT,
        )

        return ConsolidatedLog(access_log, predict_log)

    def test_constructor(self):
        consolidated_log = self.construct_consolidated_log()

        assert len(consolidated_log.logging_instance) == 2
        assert isinstance(consolidated_log.logging_instance[0], OCILog)
        assert (
            consolidated_log.logging_instance[0].compartment_id == self.COMPARTMENT_ID
        )
        assert consolidated_log.logging_instance[0].id == self.ACCESS_LOG_ID
        assert consolidated_log.logging_instance[0].log_group_id == self.LOG_GROUP_ID
        assert (
            consolidated_log.logging_instance[0].annotation
            == ModelDeploymentLogType.ACCESS
        )
        assert consolidated_log.logging_instance[0].source == self.MODEL_DEPLOYMENT_ID

        assert isinstance(consolidated_log.logging_instance[1], OCILog)
        assert (
            consolidated_log.logging_instance[1].compartment_id == self.COMPARTMENT_ID
        )
        assert consolidated_log.logging_instance[1].id == self.PREDICT_LOG_ID
        assert consolidated_log.logging_instance[1].log_group_id == self.LOG_GROUP_ID
        assert (
            consolidated_log.logging_instance[1].annotation
            == ModelDeploymentLogType.PREDICT
        )
        assert consolidated_log.logging_instance[1].source == self.MODEL_DEPLOYMENT_ID

    def test_constructor_fail(self):
        unrecognized_parameter = "unrecognized_parameter"

        with pytest.raises(
            ValueError,
            match="Unrecognized type. ConsolidatedLog constructor requires OCILog instances as parameters.",
        ):
            consolidated_log = ConsolidatedLog(unrecognized_parameter)
            assert len(consolidated_log.logging_instance) == 0

    def test_stream(self):
        consolidated_log = self.construct_consolidated_log()

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        consolidated_log.stream(
            interval=1,
            time_start=datetime.datetime.utcnow(),
            stop_condition=StopCounter().check,
        )
        sys.stdout = sys.__stdout__
        stdout = capturedOutput.getvalue()
        assert "[P] - predict log, [A] - access log\n" in stdout

    def test_tail(self):
        consolidated_log = self.construct_consolidated_log()
        tail_length = 10
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        consolidated_log.tail(
            limit=tail_length,
        )
        sys.stdout = sys.__stdout__
        stdout = capturedOutput.getvalue()
        assert "[P] - predict log, [A] - access log\n" in stdout
        assert (
            len(stdout.split("\n")) <= tail_length + 2
        )  # first log annotation and last empty element

    def test_head(self):
        consolidated_log = self.construct_consolidated_log()
        head_length = 10
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        consolidated_log.head(
            limit=head_length,
        )
        sys.stdout = sys.__stdout__
        stdout = capturedOutput.getvalue()
        assert "[P] - predict log, [A] - access log\n" in stdout
        assert (
            len(stdout.split("\n")) <= head_length + 2
        )  # first log annotation and last empty element

    def test_search(self):
        consolidated_log = self.construct_consolidated_log()
        search_length = 10
        search = consolidated_log.search(limit=search_length)
        assert len(search) <= search_length

        for i in search:
            assert isinstance(i, oci.loggingsearch.models.SearchResult)
