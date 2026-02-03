#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import unittest
from unittest.mock import MagicMock, patch

from tornado.web import HTTPError

from ads.aqua.extension.prediction_streaming_ws_msg_handler import (
    AquaPredictionStreamingWSMsgHandler,
)
from ads.aqua.common.enums import PredictEndpoints
from ads.aqua.extension.models.ws_models import RequestResponseType


class TestAquaPredictionStreamingWSMsgHandler(unittest.TestCase):
    def setUp(self):
        self.message = json.dumps(
            {
                "message_id": "msg-1",
                "kind": "PredictionStream",
                "model_deployment_id": "ocid.test",
                "params": {
                    "prompt": "hello",
                    "model": "test-model",
                    "endpoint_type": PredictEndpoints.TEXT_COMPLETIONS_ENDPOINT,
                },
            }
        )
        self.handler = AquaPredictionStreamingWSMsgHandler(self.message)

    def test_extract_text_from_choice_dict_delta(self):
        choice = {"delta": {"content": "hello"}}
        result = self.handler._extract_text_from_choice(choice)
        self.assertEqual(result, "hello")

    def test_extract_text_from_choice_dict_message(self):
        choice = {"message": {"content": "hi"}}
        result = self.handler._extract_text_from_choice(choice)
        self.assertEqual(result, "hi")

    def test_extract_text_from_choice_fallback(self):
        choice = {"text": "fallback"}
        result = self.handler._extract_text_from_choice(choice)
        self.assertEqual(result, "fallback")

    def test_extract_text_from_chunk_with_choices(self):
        chunk = {"choices": [{"delta": {"content": "chunk-text"}}]}
        result = self.handler._extract_text_from_chunk(chunk)
        self.assertEqual(result, "chunk-text")

    def test_extract_text_from_chunk_empty(self):
        self.assertIsNone(self.handler._extract_text_from_chunk(None))

    @patch("ads.aqua.extension.prediction_streaming_ws_msg_handler.AquaDeploymentApp")
    @patch("ads.aqua.extension.prediction_streaming_ws_msg_handler.OpenAI")
    def test_text_completions_streaming_success(
        self, openai_mock: MagicMock, deployment_app_mock: MagicMock
    ):
        deployment = MagicMock()
        deployment.endpoint = "http://endpoint"
        deployment_app_mock.return_value.get.return_value = deployment

        chunk = {"choices": [{"text": "hello"}]}
        openai_instance = openai_mock.return_value
        openai_instance.completions.create.return_value = [chunk]

        payload = {
            "prompt": "hello",
            "model": "test-model",
            "endpoint_type": PredictEndpoints.TEXT_COMPLETIONS_ENDPOINT,
        }

        result = list(self.handler._get_model_deployment_response("ocid.test", payload))
        self.assertEqual(result, ["hello"])

    @patch.object(AquaPredictionStreamingWSMsgHandler, "_get_model_deployment_response")
    def test_process_streaming_success(self, response_mock: MagicMock):
        response_mock.return_value = iter(["hello", "world"])

        collected = []

        def callback(resp):
            collected.append(resp.data)

        self.handler.set_stream_callback(callback)
        response = self.handler.process()

        self.assertEqual(collected, ["hello", "world"])
        self.assertEqual(response.data, "[DONE]")
        self.assertEqual(response.kind, RequestResponseType.PredictionStream)

    def test_get_message_types(self):
        types_ = AquaPredictionStreamingWSMsgHandler.get_message_types()
        self.assertEqual(types_, [RequestResponseType.PredictionStream])
