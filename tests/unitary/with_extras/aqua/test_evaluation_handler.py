#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import unittest
from unittest.mock import MagicMock, patch

from notebook.base.handlers import IPythonHandler
from parameterized import parameterized

from ads.aqua.evaluation import AquaEvaluationApp, CreateAquaEvaluationDetails
from ads.aqua.extension.base_handler import Errors
from ads.aqua.extension.evaluation_handler import (
    AquaEvaluationConfigHandler,
    AquaEvaluationHandler,
    AquaEvaluationMetricsHandler,
    AquaEvaluationReportHandler,
    AquaEvaluationStatusHandler,
)


class TestDataset:
    EVAL_ID = "ocid.datasciencemdoel.<ocid>"
    mock_valid_input = dict(
        evaluation_source_id="ocid1.datasciencemodel.oc1.iad.<OCID>",
        evaluation_name="test_evaluation_name",
        dataset_path="oci://dataset_bucket@namespace/prefix/dataset.jsonl",
        report_path="oci://report_bucket@namespace/prefix/",
        model_parameters=dict(max_token=500),
        shape_name="VM.Standard.E3.Flex",
        block_storage_size=1,
        experiment_name="test_experiment_name",
        memory_in_gbs=1,
        ocpus=1,
    )
    mock_invalid_input = dict(name="myvalue")

    def mock_url(self, action):
        return f"{self.EVAL_ID}/{action}"


class TestEvaluationHandler(unittest.TestCase):
    """Contains unittest for Evaluation handler."""

    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.test_instance = AquaEvaluationHandler(MagicMock(), MagicMock())
        self.test_instance.finish = MagicMock()

    @patch.object(AquaEvaluationApp, "delete")
    def test_delete(self, mock_delete):
        """Tests DELETE method."""
        self.test_instance.delete(TestDataset.EVAL_ID)

        self.test_instance.finish.assert_called_with(mock_delete.return_value)
        mock_delete.assert_called_with(TestDataset.EVAL_ID)

    @patch.object(AquaEvaluationApp, "cancel")
    def test_put(self, mock_cancel):
        """Tests PUT method."""
        arg = TestDataset().mock_url("status")
        self.test_instance.put(arg)

        self.test_instance.finish.assert_called_with(mock_cancel.return_value)
        mock_cancel.assert_called_with(TestDataset.EVAL_ID)

    @patch.object(AquaEvaluationApp, "create")
    def test_post(self, mock_create):
        """Tests POST method."""
        self.test_instance.get_json_body = MagicMock(
            return_value=TestDataset.mock_valid_input
        )

        self.test_instance.post()

        self.test_instance.finish.assert_called_with(mock_create.return_value)
        mock_create.assert_called_with(
            create_aqua_evaluation_details=(
                CreateAquaEvaluationDetails(**TestDataset.mock_valid_input)
            )
        )

    @parameterized.expand(
        [
            (
                dict(return_value=TestDataset.mock_invalid_input),
                400,
                "Missing required parameter:",
            ),
            (dict(side_effect=Exception()), 400, Errors.INVALID_INPUT_DATA_FORMAT),
            (dict(return_value=None), 400, Errors.NO_INPUT_DATA),
        ]
    )
    @unittest.skip(
        "Need a fix in `handle_exceptions` decorator before enabling this test."
    )
    def test_post_fail(
        self, mock_get_json_body_response, expected_status_code, expected_error_msg
    ):
        """Tests POST when encounter error."""
        self.test_instance.get_json_body = MagicMock(
            side_effect=mock_get_json_body_response.get("side_effect", None),
            return_value=mock_get_json_body_response.get("return_value", None),
        )
        self.test_instance.write_error = MagicMock()

        self.test_instance.post()

        assert (
            self.test_instance.write_error.call_args[1].get("status_code")
            == expected_status_code
        ), "Raised wrong status code."
        assert expected_error_msg in self.test_instance.write_error.call_args[1].get(
            "reason"
        ), "Error message is incorrect."

    @parameterized.expand(
        [
            ("", TestDataset.EVAL_ID, "get"),
            ("", "", "list"),
            ("aqua/evaluation/metrics", "", "get_supported_metrics"),
        ]
    )
    def test_get(self, path, input, mock_method):
        """Tests GET method."""
        self.test_instance.request = MagicMock(path=path)
        with patch.object(AquaEvaluationApp, mock_method) as mock_app_method:
            self.test_instance.get(input)

            self.test_instance.finish.assert_called_with(mock_app_method.return_value)
            mock_app_method.assert_called_once()


class TestEvaluationStatusHandler(unittest.TestCase):
    """Contains unittest for AquaEvaluationStatusHandler."""

    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.test_instance = AquaEvaluationStatusHandler(MagicMock(), MagicMock())
        self.test_instance.finish = MagicMock()

    @parameterized.expand(
        [
            (TestDataset.EVAL_ID, TestDataset.EVAL_ID),
            (TestDataset().mock_url("status"), TestDataset.EVAL_ID),
        ]
    )
    @patch.object(AquaEvaluationApp, "get_status")
    def test_get(self, input, expected_call, mock_get_status):
        """Tests GET method."""
        self.test_instance.get(input)

        self.test_instance.finish.assert_called_with(mock_get_status.return_value)
        mock_get_status.assert_called_with(expected_call)


class TestEvaluationReportHandler(unittest.TestCase):
    """Contains unittest for AquaEvaluationReportHandler."""

    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.test_instance = AquaEvaluationReportHandler(MagicMock(), MagicMock())
        self.test_instance.finish = MagicMock()

    @parameterized.expand(
        [
            (TestDataset.EVAL_ID, TestDataset.EVAL_ID),
            (TestDataset().mock_url("report"), TestDataset.EVAL_ID),
        ]
    )
    @patch.object(AquaEvaluationApp, "download_report")
    def test_get(self, input, expected_call, mock_download_report):
        """Tests GET method."""
        self.test_instance.get(input)

        self.test_instance.finish.assert_called_with(mock_download_report.return_value)
        mock_download_report.assert_called_with(expected_call)


class TestEvaluationMetricsHandler(unittest.TestCase):
    """Contains unittest for AquaEvaluationMetricsHandler."""

    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.test_instance = AquaEvaluationMetricsHandler(MagicMock(), MagicMock())
        self.test_instance.finish = MagicMock()

    @parameterized.expand(
        [
            (TestDataset.EVAL_ID, TestDataset.EVAL_ID),
            (TestDataset().mock_url("metrics"), TestDataset.EVAL_ID),
        ]
    )
    @patch.object(AquaEvaluationApp, "load_metrics")
    def test_get(self, input, expected_call, mock_load_metrics):
        """Tests GET method."""
        self.test_instance.get(input)

        self.test_instance.finish.assert_called_with(mock_load_metrics.return_value)
        mock_load_metrics.assert_called_with(expected_call)


class TestEvaluationConfigHandler(unittest.TestCase):
    """Contains unittest for AquaEvaluationConfigHandler."""

    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.test_instance = AquaEvaluationConfigHandler(MagicMock(), MagicMock())
        self.test_instance.finish = MagicMock()

    @patch.object(AquaEvaluationApp, "load_evaluation_config")
    def test_get(self, mock_load_evaluation_config):
        """Tests GET method."""
        self.test_instance.get(TestDataset.EVAL_ID)

        self.test_instance.finish.assert_called_with(
            mock_load_evaluation_config.return_value
        )
        mock_load_evaluation_config.assert_called_with(TestDataset.EVAL_ID)
