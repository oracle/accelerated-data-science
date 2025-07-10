#!/usr/bin/env python
# -*- coding: utf-8 -*--
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from unicodedata import category
from unittest import TestCase
from unittest.mock import MagicMock, patch, ANY

import pytest
from huggingface_hub.hf_api import HfApi, ModelInfo
from huggingface_hub.utils import GatedRepoError
from notebook.base.handlers import HTTPError, IPythonHandler
from parameterized import parameterized

from ads.aqua.common.errors import AquaRuntimeError
from ads.aqua.common.utils import get_hf_model_info
from ads.aqua.constants import AQUA_TROUBLESHOOTING_LINK, STATUS_CODE_MESSAGES, AQUA_CHAT_TEMPLATE_METADATA_KEY
from ads.aqua.extension.errors import ReplyDetails
from ads.aqua.extension.model_handler import (
    AquaHuggingFaceHandler,
    AquaModelHandler,
    AquaModelLicenseHandler,
    AquaModelChatTemplateHandler
)
from ads.aqua.model import AquaModelApp
from ads.aqua.model.entities import AquaModel, AquaModelSummary, HFModelSummary
from ads.config import SERVICE, USER


class ModelHandlerTestCase(TestCase):
    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.model_handler = AquaModelHandler(MagicMock(), MagicMock())
        self.model_handler.request = MagicMock()

    def tearDown(self) -> None:
        get_hf_model_info.cache_clear()

    @patch("ads.aqua.extension.model_handler.urlparse")
    @patch.object(AquaModelHandler, "list")
    def test_get_no_id(self, mock_list, mock_urlparse):
        request_path = MagicMock(path="aqua/model")
        mock_urlparse.return_value = request_path
        self.model_handler.get()
        mock_list.assert_called()

    @patch("ads.aqua.extension.model_handler.urlparse")
    @patch.object(AquaModelHandler, "read")
    def test_get_with_id(self, mock_read, mock_urlparse):
        request_path = MagicMock(path="aqua/model/test_model_id")
        mock_urlparse.return_value = request_path
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

    @patch("ads.aqua.extension.model_handler.urlparse")
    @patch.object(AquaModelApp, "delete_model")
    def test_delete_with_id(self, mock_delete, mock_urlparse):
        request_path = MagicMock(path="aqua/model/ocid1.datasciencemodel.oc1.iad.xxx")
        mock_urlparse.return_value = request_path
        mock_delete.return_value = {"state": "DELETED"}
        with patch(
            "ads.aqua.extension.base_handler.AquaAPIhandler.finish"
        ) as mock_finish:
            mock_finish.side_effect = lambda x: x
            result = self.model_handler.delete(id="ocid1.datasciencemodel.oc1.iad.xxx")
            assert result["state"] is "DELETED"
            mock_urlparse.assert_called()
            mock_delete.assert_called()

    @patch.object(AquaModelApp, "list_valid_inference_containers")
    @patch.object(AquaModelApp, "edit_registered_model")
    def test_put(self, mock_edit, mock_inference_container_list):
        mock_edit.return_value = None
        mock_inference_container_list.return_value = [
            "odsc-vllm-serving",
            "odsc-tgi-serving",
            "odsc-llama-cpp-serving",
        ]
        self.model_handler.get_json_body = MagicMock(
            return_value=dict(
                task="text_generation",
                enable_finetuning="true",
                inference_container="odsc-tgi-serving",
            )
        )
        with patch(
            "ads.aqua.extension.base_handler.AquaAPIhandler.finish"
        ) as mock_finish:
            mock_finish.side_effect = lambda x: x
            result = self.model_handler.put(id="ocid1.datasciencemodel.oc1.iad.xxx")
            assert result is None
            mock_edit.assert_called_once()
            mock_inference_container_list.assert_called_once()

    @patch.object(AquaModelApp, "list")
    def test_list(self, mock_list):
        with patch(
            "ads.aqua.extension.base_handler.AquaAPIhandler.finish"
        ) as mock_finish:
            mock_finish.side_effect = lambda x: x
            self.model_handler.list()
            mock_list.assert_called_with(
                compartment_id=None, project_id=None, model_type=None, category=SERVICE
            )

    @parameterized.expand(
        [
            (None, None, False, None, None, None, None, None, True),
            (
                "odsc-llm-fine-tuning",
                None,
                False,
                None,
                None,
                ["test.json"],
                None,
                None,
                False,
            ),
            (None, "test.gguf", True, None, ["*.json"], None, None, None, False),
            (
                None,
                None,
                True,
                "iad.ocir.io/<namespace>/<image>:<tag>",
                ["*.json"],
                ["test.json"],
                None,
                None,
                False,
            ),
            (
                None,
                None,
                False,
                None,
                None,
                None,
                {"ftag1": "fvalue1"},
                {"dtag1": "dvalue1"},
                False,
            ),
        ],
    )
    @patch("notebook.base.handlers.APIHandler.finish")
    @patch("ads.aqua.model.AquaModelApp.register")
    def test_register(
        self,
        finetuning_container,
        model_file,
        download_from_hf,
        inference_container_uri,
        allow_patterns,
        ignore_patterns,
        freeform_tags,
        defined_tags,
        ignore_model_artifact_check,
        mock_register,
        mock_finish,
    ):
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
                finetuning_container=finetuning_container,
                model_file=model_file,
                download_from_hf=download_from_hf,
                inference_container_uri=inference_container_uri,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                freeform_tags=freeform_tags,
                defined_tags=defined_tags,
                ignore_model_artifact_check=ignore_model_artifact_check,
            )
        )
        result = self.model_handler.post()
        mock_register.assert_called_with(
            model="test_model_name",
            os_path="test_os_path",
            inference_container="odsc-tgi-serving",
            finetuning_container=finetuning_container,
            compartment_id=None,
            project_id=None,
            model_file=model_file,
            download_from_hf=download_from_hf,
            local_dir=None,
            cleanup_model_cache=False,
            inference_container_uri=inference_container_uri,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            freeform_tags=freeform_tags,
            defined_tags=defined_tags,
            ignore_model_artifact_check=ignore_model_artifact_check,
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


class AquaModelChatTemplateHandlerTestCase(TestCase):
    @patch.object(IPythonHandler, "__init__")
    def setUp(self, ipython_init_mock) -> None:
        ipython_init_mock.return_value = None
        self.model_chat_template_handler = AquaModelChatTemplateHandler(
            MagicMock(), MagicMock()
        )
        self.model_chat_template_handler.finish = MagicMock()
        self.model_chat_template_handler.request = MagicMock()
        self.model_chat_template_handler._headers = {}

    @patch("ads.aqua.extension.model_handler.OCIDataScienceModel.from_id")
    @patch("ads.aqua.extension.model_handler.urlparse")
    def test_get_valid_path(self, mock_urlparse, mock_from_id):
        request_path = MagicMock(path="/aqua/models/ocid1.xx./chat-template")
        mock_urlparse.return_value = request_path

        model_mock = MagicMock()
        model_mock.get_custom_metadata_artifact.return_value = "chat_template_string"
        mock_from_id.return_value = model_mock

        self.model_chat_template_handler.get(model_id="test_model_id")
        self.model_chat_template_handler.finish.assert_called_with("chat_template_string")
        model_mock.get_custom_metadata_artifact.assert_called_with("chat_template")

    @patch("ads.aqua.extension.model_handler.urlparse")
    def test_get_invalid_path(self, mock_urlparse):
        request_path = MagicMock(path="/wrong/path")
        mock_urlparse.return_value = request_path

        with self.assertRaises(HTTPError) as context:
            self.model_chat_template_handler.get("ocid1.test.chat")
        self.assertEqual(context.exception.status_code, 400)

    @patch("ads.aqua.extension.model_handler.OCIDataScienceModel.from_id", side_effect=Exception("Not found"))
    @patch("ads.aqua.extension.model_handler.urlparse")
    def test_get_model_not_found(self, mock_urlparse, mock_from_id):
        request_path = MagicMock(path="/aqua/models/ocid1.invalid/chat-template")
        mock_urlparse.return_value = request_path

        with self.assertRaises(HTTPError) as context:
            self.model_chat_template_handler.get("ocid1.invalid")
        self.assertEqual(context.exception.status_code, 404)

    @patch("ads.aqua.extension.model_handler.DataScienceModel.from_id")
    def test_post_valid(self, mock_from_id):
        model_mock = MagicMock()
        model_mock.create_custom_metadata_artifact.return_value = {"result": "success"}
        mock_from_id.return_value = model_mock

        self.model_chat_template_handler.get_json_body = MagicMock(return_value={"chat_template": "Hello <|user|>"})
        result = self.model_chat_template_handler.post("ocid1.valid")
        self.model_chat_template_handler.finish.assert_called_with({"result": "success"})

        model_mock.create_custom_metadata_artifact.assert_called_with(
            metadata_key_name=AQUA_CHAT_TEMPLATE_METADATA_KEY,
            path_type=ANY,
            artifact_path_or_content=b"Hello <|user|>"
        )

    @patch.object(AquaModelChatTemplateHandler, "write_error")
    def test_post_invalid_json(self, mock_write_error):
        self.model_chat_template_handler.get_json_body = MagicMock(side_effect=Exception("Invalid JSON"))
        self.model_chat_template_handler._headers = {}
        self.model_chat_template_handler.post("ocid1.test.invalidjson")

        mock_write_error.assert_called_once()

        kwargs = mock_write_error.call_args.kwargs
        exc_info = kwargs.get("exc_info")

        assert exc_info is not None
        exc_type, exc_instance, _ = exc_info

        assert isinstance(exc_instance, HTTPError)
        assert exc_instance.status_code == 400
        assert "Invalid JSON body" in str(exc_instance)

    @patch.object(AquaModelChatTemplateHandler, "write_error")
    def test_post_missing_chat_template(self, mock_write_error):
        self.model_chat_template_handler.get_json_body = MagicMock(return_value={})
        self.model_chat_template_handler._headers = {}

        self.model_chat_template_handler.post("ocid1.test.model")

        mock_write_error.assert_called_once()
        exc_info = mock_write_error.call_args.kwargs.get("exc_info")
        assert exc_info is not None
        _, exc_instance, _ = exc_info
        assert isinstance(exc_instance, HTTPError)
        assert exc_instance.status_code == 400
        assert "Missing required field: 'chat_template'" in str(exc_instance)

    @patch("ads.aqua.extension.model_handler.DataScienceModel.from_id", side_effect=Exception("Not found"))
    @patch.object(AquaModelChatTemplateHandler, "write_error")
    def test_post_model_not_found(self, mock_write_error, mock_from_id):
        self.model_chat_template_handler.get_json_body = MagicMock(return_value={"chat_template": "test template"})
        self.model_chat_template_handler._headers = {}

        self.model_chat_template_handler.post("ocid1.invalid.model")

        mock_write_error.assert_called_once()
        exc_info = mock_write_error.call_args.kwargs.get("exc_info")
        assert exc_info is not None
        _, exc_instance, _ = exc_info
        assert isinstance(exc_instance, HTTPError)
        assert exc_instance.status_code == 404
        assert "Model not found" in str(exc_instance)


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
                mock_get_model.assert_called_with(expected_aqua_model_id)
            else:
                assert test_result == None

    @patch("ads.aqua.common.utils.format_hf_custom_error_message")
    @patch("uuid.uuid4")
    def test_post_negative(self, mock_uuid, mock_format_hf_custom_error_message):
        mock_uuid.return_value = "###"

        # case 1
        self.mock_handler.get_json_body = MagicMock(side_effect=ValueError())
        self.mock_handler.post()
        self.mock_handler.finish.assert_called_with(
            ReplyDetails(
                status=400,
                troubleshooting_tips=f"For general tips on troubleshooting: {AQUA_TROUBLESHOOTING_LINK}",
                message="Invalid format of input data.",
                service_payload={},
                reason="Invalid format of input data.",
                request_id="###",
            )
        )
        get_hf_model_info.cache_clear()

        # case 2
        self.mock_handler.get_json_body = MagicMock(return_value={})
        self.mock_handler.post()
        self.mock_handler.finish.assert_called_with(
            ReplyDetails(
                status=400,
                troubleshooting_tips=f"For general tips on troubleshooting: {AQUA_TROUBLESHOOTING_LINK}",
                message="No input data provided.",
                service_payload={},
                reason="No input data provided.",
                request_id="###",
            )
        )
        get_hf_model_info.cache_clear()

        # case 3
        self.mock_handler.get_json_body = MagicMock(return_value={"some_field": None})
        self.mock_handler.post()
        self.mock_handler.finish.assert_called_with(
            ReplyDetails(
                status=400,
                troubleshooting_tips=f"For general tips on troubleshooting: {AQUA_TROUBLESHOOTING_LINK}",
                message="Missing required parameter: 'model_id'",
                service_payload={},
                reason="Missing required parameter: 'model_id'",
                request_id="###",
            )
        )
        get_hf_model_info.cache_clear()

        # case 4
        self.mock_handler.get_json_body = MagicMock(
            return_value={"model_id": "test_model_id"}
        )
        mock_format_hf_custom_error_message.side_effect = AquaRuntimeError(
            "test error message"
        )

        with patch.object(HfApi, "model_info") as mock_model_info:
            mock_model_info.side_effect = GatedRepoError(message="test message")
            self.mock_handler.post()
            self.mock_handler.finish.assert_called_with(
                ReplyDetails(
                    status=400,
                    troubleshooting_tips=f"For general tips on troubleshooting: {AQUA_TROUBLESHOOTING_LINK}",
                    message=STATUS_CODE_MESSAGES["400"],
                    service_payload={},
                    reason="test error message",
                    request_id="###",
                )
            )
        get_hf_model_info.cache_clear()

        # case 5
        self.mock_handler.get_json_body = MagicMock(
            return_value={"model_id": "test_model_id"}
        )
        with patch.object(HfApi, "model_info") as mock_model_info:
            mock_model_info.return_value = MagicMock(disabled=True, id="test_model_id")
            self.mock_handler.post()

            self.mock_handler.finish.assert_called_with(
                ReplyDetails(
                    status=400,
                    troubleshooting_tips=f"For general tips on troubleshooting: {AQUA_TROUBLESHOOTING_LINK}",
                    message=STATUS_CODE_MESSAGES["400"],
                    service_payload={},
                    reason="The chosen model 'test_model_id' is currently disabled and cannot be imported into AQUA. Please verify the model's status on the Hugging Face Model Hub or select a different model.",
                    request_id="###",
                )
            )
        get_hf_model_info.cache_clear()

        # # case 6 pipeline Tag
        # self.mock_handler.get_json_body = MagicMock(
        #     return_value={"model_id": "test_model_id"}
        # )
        # with patch.object(HfApi, "model_info") as mock_model_info:
        #     mock_model_info.return_value = MagicMock(
        #         disabled=False, id="test_model_id", pipeline_tag="not-text-generation"
        #     )
        #     self.mock_handler.post()
        #     self.mock_handler.finish.assert_called_with(
        #         '{"status": 400, "message": "Something went wrong with your request.", '
        #         '"service_payload": {}, "reason": "Unsupported pipeline tag for the chosen '
        #         "model: 'not-text-generation'. AQUA currently supports the following tasks only: "
        #         f'{", ".join(ModelTask.values())}. '
        #         'Please select a model with a compatible pipeline tag.", "request_id": "###"}'
        #     )
        get_hf_model_info.cache_clear()

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
            test_hf_model_info = ModelInfo(
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
