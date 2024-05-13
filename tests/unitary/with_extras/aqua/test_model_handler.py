#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest import TestCase
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import HfApi, hf_api
from huggingface_hub.utils import GatedRepoError
from notebook.base.handlers import IPythonHandler
from tornado.web import HTTPError

from ads.aqua.extension.model_handler import (
    AquaHuggingFaceHandler,
    AquaModelHandler,
    AquaModelLicenseHandler,
)
from ads.aqua.model import AquaModelApp
from ads.aqua.model.entities import AquaModelSummary, HFModelSummary


class ModelHandlerTestCase(TestCase):
    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.model_handler = AquaModelHandler(MagicMock(), MagicMock())
        self.model_handler.request = MagicMock()
        self.model_handler.finish = MagicMock()

    @patch.object(AquaModelHandler, "list")
    def test_get_no_id(self, mock_list):
        self.model_handler.get()
        mock_list.assert_called()

    @patch.object(AquaModelHandler, "read")
    def test_get_with_id(self, mock_read):
        self.model_handler.get(model_id="test_model_id")
        mock_read.assert_called_with("test_model_id")

    @patch.object(AquaModelApp, "get")
    def test_read(self, mock_get):
        self.model_handler.read(model_id="test_model_id")
        self.model_handler.finish.assert_called_with(mock_get.return_value)
        mock_get.assert_called_with("test_model_id")

    @patch.object(AquaModelApp, "clear_model_list_cache")
    @patch("ads.aqua.extension.model_handler.urlparse")
    def test_delete(self, mock_urlparse, mock_clear_model_list_cache):
        request_path = MagicMock(path="aqua/model/cache")
        mock_urlparse.return_value = request_path

        self.model_handler.delete()
        self.model_handler.finish.assert_called_with(
            mock_clear_model_list_cache.return_value
        )

        mock_urlparse.assert_called()
        mock_clear_model_list_cache.assert_called()

    @patch.object(AquaModelApp, "list")
    def test_list(self, mock_list):
        self.model_handler.list()

        self.model_handler.finish.assert_called_with(mock_list.return_value)
        mock_list.assert_called_with(None, None, model_type=None)


class ModelLicenseHandlerTestCase(TestCase):
    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.model_license_handler = AquaModelLicenseHandler(MagicMock(), MagicMock())
        self.model_license_handler.finish = MagicMock()

    @patch.object(AquaModelApp, "load_license")
    def test_get(self, mock_load_license):
        self.model_license_handler.get(model_id="test_model_id")

        self.model_license_handler.finish.assert_called_with(
            mock_load_license.return_value
        )
        mock_load_license.assert_called_with("test_model_id")


class TestAquaHuggingFaceHandler:
    def setup_method(self):
        with patch.object(IPythonHandler, "__init__"):
            self.mock_handler = AquaHuggingFaceHandler(MagicMock(), MagicMock())
            self.mock_handler.request = MagicMock()
            self.mock_handler.finish = MagicMock()
            self.mock_handler.set_header = MagicMock()
            self.mock_handler.set_status = MagicMock()

    @pytest.mark.parametrize(
        "test_model_id, test_author, expected_aqua_model_name, expected_aqua_model_id",
        [
            ("organization1/name1", "organization1", "organization1/name1", "test_id1"),
            ("organization1/name2", "organization1", "organization1/name2", "test_id2"),
            ("organization2/name3", "organization2", "organization2/name3", "test_id3"),
            ("non_existing_name", "organization2", None, None),
            ("organization1/non_existing_name", "organization1", None, None),
        ],
    )
    @patch.object(AquaModelApp, "get")
    def test_find_matching_aqua_model(
        self,
        mock_get_model,
        test_model_id,
        test_author,
        expected_aqua_model_name,
        expected_aqua_model_id,
    ):
        with patch.object(AquaModelApp, "list") as aqua_model_mock_list:
            aqua_model_mock_list.return_value = [
                AquaModelSummary(
                    id="test_id1",
                    name="organization1/name1",
                    organization="organization1",
                ),
                AquaModelSummary(
                    id="test_id2",
                    name="organization1/name2",
                    organization="organization1",
                ),
                AquaModelSummary(
                    id="test_id3",
                    name="organization2/name3",
                    organization="organization2",
                ),
            ]

            test_result = self.mock_handler._find_matching_aqua_model(
                model_id=test_model_id
            )

            aqua_model_mock_list.assert_called_once()

            if expected_aqua_model_name:
                mock_get_model.assert_called_with(
                    expected_aqua_model_id, load_model_card=False
                )
            else:
                assert test_result == None

    @patch("uuid.uuid4")
    def test_post_negative(self, mock_uuid):
        mock_uuid.return_value = "###"

        # case 1
        self.mock_handler.get_json_body = MagicMock(side_effect=ValueError())
        self.mock_handler.post()
        self.mock_handler.finish.assert_called_with(
            '{"status": 400, "message": "Invalid format of input data.", "service_payload": {}, "reason": "Invalid format of input data.", "request_id": "###"}'
        )

        # case 2
        self.mock_handler.get_json_body = MagicMock(return_value={})
        self.mock_handler.post()
        self.mock_handler.finish.assert_called_with(
            '{"status": 400, "message": "No input data provided.", "service_payload": {}, '
            '"reason": "No input data provided.", "request_id": "###"}'
        )

        # case 3
        self.mock_handler.get_json_body = MagicMock(return_value={"some_field": None})
        self.mock_handler.post()
        self.mock_handler.finish.assert_called_with(
            '{"status": 400, "message": "Missing required parameter: \'model_id\'", '
            '"service_payload": {}, "reason": "Missing required parameter: \'model_id\'", "request_id": "###"}'
        )

        # case 4
        self.mock_handler.get_json_body = MagicMock(
            return_value={"model_id": "test_model_id"}
        )
        self.mock_handler._format_custom_error_message = MagicMock(
            return_value="test error message"
        )
        with patch.object(HfApi, "model_info") as mock_model_info:
            mock_model_info.side_effect = GatedRepoError(message="test message")
            self.mock_handler.post()
            self.mock_handler.finish.assert_called_with(
                '{"status": 400, "message": "Something went wrong with your request.", '
                '"service_payload": {}, "reason": "test error message", "request_id": "###"}'
            )

        # case 5
        self.mock_handler.get_json_body = MagicMock(
            return_value={"model_id": "test_model_id"}
        )
        with patch.object(HfApi, "model_info") as mock_model_info:
            mock_model_info.return_value = MagicMock(disabled=True, id="test_model_id")
            self.mock_handler.post()
            self.mock_handler.finish.assert_called_with(
                '{"status": 400, "message": "Something went wrong with your request.", "service_payload": {}, '
                '"reason": "The chosen model \'test_model_id\' is currently disabled and cannot be '
                "imported into AQUA. Please verify the model's status on the Hugging Face Model "
                'Hub or select a different model.", "request_id": "###"}'
            )

        # case 6
        self.mock_handler.get_json_body = MagicMock(
            return_value={"model_id": "test_model_id"}
        )
        with patch.object(HfApi, "model_info") as mock_model_info:
            mock_model_info.return_value = MagicMock(
                disabled=False, id="test_model_id", pipeline_tag="not-text-generation"
            )
            self.mock_handler.post()
            self.mock_handler.finish.assert_called_with(
                '{"status": 400, "message": "Something went wrong with your request.", '
                '"service_payload": {}, "reason": "Unsupported pipeline tag for the chosen '
                "model: 'not-text-generation'. AQUA currently supports the following tasks only: "
                'text-generation. Please select a model with a compatible pipeline tag.", "request_id": "###"}'
            )

    @patch("uuid.uuid4")
    def test_post_positive(self, mock_uuid):
        mock_uuid.return_value = "###"

        self.mock_handler.get_json_body = MagicMock(
            return_value={"model_id": "test_model_id"}
        )

        test_aqua_model_summary = AquaModelSummary(
            name="name1", organization="organization1"
        )
        self.mock_handler._find_matching_aqua_model = MagicMock(
            return_value=test_aqua_model_summary
        )

        with patch.object(HfApi, "model_info") as mock_model_info:
            test_hf_model_info = hf_api.ModelInfo(
                disabled=False,
                id="test_model_id",
                pipeline_tag="text-generation",
                author="test_author",
                private=False,
                downloads=10,
                likes=10,
                tags=None,
            )
            mock_model_info.return_value = test_hf_model_info
            self.mock_handler.post()

            self.mock_handler._find_matching_aqua_model.assert_called_with(
                model_id="test_model_id"
            )

            test_model_summary = HFModelSummary(
                model_info=test_hf_model_info, aqua_model_info=test_aqua_model_summary
            )

            self.mock_handler.finish.assert_called_with(test_model_summary)
