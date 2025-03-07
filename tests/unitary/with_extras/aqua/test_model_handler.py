#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest import TestCase
from unittest.mock import MagicMock, patch

from notebook.base.handlers import IPythonHandler
from ads.aqua.extension.model_handler import (
    AquaModelHandler,
    AquaModelLicenseHandler,
)
from ads.aqua.model import AquaModelApp
from ads.aqua.model.entities import AquaModel


class ModelHandlerTestCase(TestCase):
    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.model_handler = AquaModelHandler(MagicMock(), MagicMock())
        self.model_handler.request = MagicMock()

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
        with patch(
            "ads.aqua.extension.base_handler.AquaAPIhandler.finish"
        ) as mock_finish:
            mock_finish.side_effect = lambda x: x
            self.model_handler.read(model_id="test_model_id")
            mock_get.assert_called_with("test_model_id")

    @patch.object(AquaModelApp, "clear_model_list_cache")
    @patch("ads.aqua.extension.model_handler.urlparse")
    def test_delete(self, mock_urlparse, mock_clear_model_list_cache):
        request_path = MagicMock(path="aqua/model/cache")
        mock_urlparse.return_value = request_path
        mock_clear_model_list_cache.return_value = {
            "key": {
                "compartment_id": "test-compartment-ocid",
            },
            "cache_deleted": True,
        }

        with patch(
            "ads.aqua.extension.base_handler.AquaAPIhandler.finish"
        ) as mock_finish:
            mock_finish.side_effect = lambda x: x
            result = self.model_handler.delete()
            assert result["cache_deleted"] is True
            mock_urlparse.assert_called()
            mock_clear_model_list_cache.assert_called()

    @patch.object(AquaModelApp, "list")
    def test_list(self, mock_list):
        with patch(
            "ads.aqua.extension.base_handler.AquaAPIhandler.finish"
        ) as mock_finish:
            mock_finish.side_effect = lambda x: x
            self.model_handler.list()
            mock_list.assert_called_with(
                compartment_id=None, project_id=None, model_type=None
            )

    @patch("notebook.base.handlers.APIHandler.finish")
    @patch("ads.aqua.model.AquaModelApp.register")
    def test_register(self, mock_register, mock_finish):
        mock_register.return_value = AquaModel(
            id="test_id",
            inference_container="odsc-tgi-serving",
            evaluation_container="odsc-llm-evaluate",
        )
        mock_finish.side_effect = lambda x: x

        self.model_handler.get_json_body = MagicMock(
            return_value=dict(
                model="test_model_name",
                os_path="test_os_path",
                inference_container="odsc-tgi-serving",
            )
        )
        result = self.model_handler.post()
        mock_register.assert_called_with(
            model="test_model_name",
            os_path="test_os_path",
            inference_container="odsc-tgi-serving",
            finetuning_container=None,
            compartment_id=None,
            project_id=None,
        )
        assert result["id"] == "test_id"
        assert result["inference_container"] == "odsc-tgi-serving"
        assert result["evaluation_container"] == "odsc-llm-evaluate"
        assert result["finetuning_container"] is None


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
