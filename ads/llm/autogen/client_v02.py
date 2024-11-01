# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.

"""This module contains the LangChain LLM client for AutoGen
# References:
# https://microsoft.github.io/autogen/0.2/docs/notebooks/agentchat_huggingface_langchain/
# https://github.com/microsoft/autogen/blob/0.2/notebook/agentchat_custom_model.ipynb
"""
import copy
import importlib
import logging
from typing import Dict, List, Union
from types import SimpleNamespace


from autogen import ModelClient
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


class Message(AIMessage):
    """Represents message returned from the LLM."""

    @classmethod
    def from_message(cls, message: AIMessage):
        """Converts from LangChain AIMessage."""
        message = copy.deepcopy(message)
        message.__class__ = cls
        return message

    @property
    def function_call(self):
        """Function calls."""
        return self.tool_calls


class LangChainModelClient(ModelClient):
    """Represents a model client wrapping a LangChain chat model."""

    def __init__(self, config: dict, **kwargs) -> None:
        super().__init__()
        logger.info("LangChain model client config: %s", str(config))
        self.client_class = config.pop("model_client_cls")
        # Parameters for the model
        self.model_name = config.get("model")
        # Import the LangChain class
        if "langchain_cls" not in config:
            raise ValueError("Missing langchain_cls in LangChain Model Client config.")
        module_cls = config.pop("langchain_cls")
        module_name, cls_name = str(module_cls).rsplit(".", 1)
        langchain_module = importlib.import_module(module_name)
        langchain_cls = getattr(langchain_module, cls_name)
        # Initialize the LangChain client
        self.model = langchain_cls(**config)

    def create(self, params) -> ModelClient.ModelClientResponseProtocol:
        streaming = params.get("stream", False)
        num_of_responses = params.get("n", 1)
        messages = params.get("messages", [])

        response = SimpleNamespace()
        response.choices = []
        response.model = self.model_name

        if streaming and messages:
            # If streaming is enabled and has messages, then iterate over the chunks of the response.
            raise NotImplementedError()
        else:
            # If streaming is not enabled, send a regular chat completion request
            ai_message = self.model.invoke(messages)
            choice = SimpleNamespace()
            choice.message = Message.from_message(ai_message)
            response.choices.append(choice)
        return response

    def message_retrieval(
        self, response: ModelClient.ModelClientResponseProtocol
    ) -> Union[List[str], List[ModelClient.ModelClientResponseProtocol.Choice.Message]]:
        """
        Retrieve and return a list of strings or a list of Choice.Message from the response.

        NOTE: if a list of Choice.Message is returned, it currently needs to contain the fields of OpenAI's ChatCompletion Message object,
        since that is expected for function or tool calling in the rest of the codebase at the moment, unless a custom agent is being used.
        """
        return [choice.message.content for choice in response.choices]

    def cost(self, response: ModelClient.ModelClientResponseProtocol) -> float:
        response.cost = 0
        return 0

    @staticmethod
    def get_usage(response: ModelClient.ModelClientResponseProtocol) -> Dict:
        """Return usage summary of the response using RESPONSE_USAGE_KEYS."""
        return {}
