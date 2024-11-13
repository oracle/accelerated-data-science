# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.
import sys
from unittest import TestCase, mock

import pytest

if sys.version_info < (3, 9):
    pytest.skip(allow_module_level=True)

import autogen
from langchain_core.messages import AIMessage, ToolCall
from ads.llm.autogen.client_v02 import (
    LangChainModelClient,
    register_custom_client,
    custom_clients,
)
from ads.llm import ChatOCIModelDeploymentVLLM


ODSC_LLM_CONFIG = {
    "model_client_cls": "LangChainModelClient",
    "langchain_cls": "ads.llm.ChatOCIModelDeploymentVLLM",
    "model": "Mistral",
    "client_params": {
        "model": "odsc-llm",
        "endpoint": "<ODSC_ENDPOINT>",
        "model_kwargs": {"temperature": 0, "max_tokens": 500},
    },
}

TEST_PAYLOAD = {
    "messages": ["hello", "hi"],
    "tool": {
        "type": "function",
        "function": {
            "name": "my_tool",
            "description": "my_desc",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The customer's order ID.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
}

MOCKED_RESPONSE_CONTENT = "hello"
MOCKED_AI_MESSAGE = AIMessage(
    content=MOCKED_RESPONSE_CONTENT,
    tool_calls=[ToolCall(name="my_tool", args={"arg": "val"}, id="a")],
)
MOCKED_TOOL_CALL = [
    {
        "id": "a",
        "function": {
            "name": "my_tool",
            "arguments": '{"arg": "val"}',
        },
        "type": "function",
    }
]


class AutoGenTestCase(TestCase):
    @mock.patch("ads.common.auth.default_signer", return_value=dict(signer=None))
    def test_register_client(self, signer):
        # There should be no custom client before registration.
        self.assertEqual(custom_clients, {})
        register_custom_client(LangChainModelClient)
        self.assertEqual(custom_clients, {"LangChainModelClient": LangChainModelClient})
        # Test LLM config without custom LLM
        config_list = [
            {
                "model": "llama-7B",
                "api_key": "123",
            }
        ]
        wrapper = autogen.oai.client.OpenAIWrapper(config_list=config_list)
        self.assertEqual(type(wrapper._clients[0]), autogen.oai.client.OpenAIClient)
        # Test LLM config with custom LLM
        config_list = [ODSC_LLM_CONFIG]
        wrapper = autogen.oai.client.OpenAIWrapper(config_list=config_list)
        self.assertEqual(type(wrapper._clients[0]), LangChainModelClient)

    @mock.patch("ads.common.auth.default_signer", return_value=dict(signer=None))
    @mock.patch(
        "ads.llm.ChatOCIModelDeploymentVLLM.invoke", return_value=MOCKED_AI_MESSAGE
    )
    def test_create_completion(self, mocked_invoke, *args):
        client = LangChainModelClient(config=ODSC_LLM_CONFIG)
        self.assertEqual(client.model_name, "Mistral")
        self.assertEqual(type(client.model), ChatOCIModelDeploymentVLLM)
        self.assertEqual(client.model._invocation_params(stop=None)["max_tokens"], 500)
        response = client.create(TEST_PAYLOAD)
        message = response.choices[0].message
        self.assertEqual(message.content, MOCKED_RESPONSE_CONTENT)
        self.assertEqual(message.tool_calls, MOCKED_TOOL_CALL)
