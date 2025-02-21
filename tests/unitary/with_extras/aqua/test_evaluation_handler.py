#!/usr/bin/env python
# -*- coding: utf-8 -*--
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import unittest
from unittest.mock import MagicMock, patch

from notebook.base.handlers import IPythonHandler
from parameterized import parameterized

from ads.aqua.evaluation import AquaEvaluationApp
from ads.aqua.evaluation.entities import CreateAquaEvaluationDetails
from ads.aqua.extension.evaluation_handler import AquaEvaluationHandler
from tests.unitary.with_extras.aqua.utils import HandlerTestDataset as TestDataset


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
        self.test_instance.delete(TestDataset.MOCK_OCID)

        self.test_instance.finish.assert_called_with(mock_delete.return_value)
        mock_delete.assert_called_with(TestDataset.MOCK_OCID)

    @patch.object(AquaEvaluationApp, "cancel")
    def test_put(self, mock_cancel):
        """Tests PUT method."""
        arg = TestDataset().mock_url("status")
        self.test_instance.put(arg)

        self.test_instance.finish.assert_called_with(mock_cancel.return_value)
        mock_cancel.assert_called_with(TestDataset.MOCK_OCID)

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
            ("", TestDataset.MOCK_OCID, "get"),
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
