# Copyright (c) 2024, Oracle and/or its affiliates.  All rights reserved.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""This module contains the custom LLM client for AutoGen v0.2 to use LangChain chat models.
https://microsoft.github.io/autogen/0.2/blog/2024/01/26/Custom-Models/

To use the custom client:
1. Prepare the LLM config, including the parameters for initializing the LangChain client.
2. Register the custom LLM

The LLM config should config the following keys:
* model_client_cls: Required by AutoGen to identify the custom client. It should be "LangChainModelClient"
* langchain_cls: LangChain class including the full import path.
* model: Name of the model to be used by AutoGen
* client_params: A dictionary containing the parameters to initialize the LangChain chat model.

Although the `LangChainModelClient` is designed to be generic and can potentially support any LangChain chat model,
the invocation depends on the server API spec and it may not be compatible with some implementations.

Following is an example config for OCI Generative AI service:
{
    "model_client_cls": "LangChainModelClient",
    "langchain_cls": "langchain_community.chat_models.oci_generative_ai.ChatOCIGenAI",
    "model": "cohere.command-r-plus",
    # client_params will be used to initialize the LangChain ChatOCIGenAI class.
    "client_params": {
        "model_id": "cohere.command-r-plus",
        "compartment_id": COMPARTMENT_OCID,
        "model_kwargs": {"temperature": 0, "max_tokens": 2048},
        # Update the authentication method as needed
        "auth_type": "SECURITY_TOKEN",
        "auth_profile": "DEFAULT",
        # You may need to specify `service_endpoint` if the service is in a different region.
    },
}

Following is an example config for OCI Data Science Model Deployment:
{
    "model_client_cls": "LangChainModelClient",
    "langchain_cls": "ads.llm.ChatOCIModelDeploymentVLLM",
    "model": "odsc-llm",
    "endpoint": "https://MODEL_DEPLOYMENT_URL/predict",
    "model_kwargs": {"temperature": 0.1, "max_tokens": 2048},
    # function_call_params will only be added to the API call when function/tools are added.
    "function_call_params": {
        "tool_choice": "auto",
        "chat_template": ChatTemplates.mistral(),
    },
}

Note that if `client_params` is not specified in the config, all arguments from the config except
`model_client_cls` and `langchain_cls`, and `function_call_params`, will be used to initialize
the LangChain chat model.

The `function_call_params` will only be used for function/tool calling when tools are specified.

To register the custom client:

from ads.llm.autogen.client_v02 import LangChainModelClient, register_custom_client
register_custom_client(LangChainModelClient)

Once registered with ADS, the custom LLM class will be auto-registered for all new agents.
There is no need to call `register_model_client()` on each agent.

References:
https://microsoft.github.io/autogen/0.2/docs/notebooks/agentchat_huggingface_langchain/
https://github.com/microsoft/autogen/blob/0.2/notebook/agentchat_custom_model.ipynb

"""
import copy
import importlib
import json
import logging
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Union

from autogen import ModelClient
from autogen.oai.client import OpenAIWrapper, PlaceHolderClient
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

# custom_clients is a dictionary mapping the name of the class to the actual class
custom_clients = {}

# There is a bug in GroupChat when using custom client:
# https://github.com/microsoft/autogen/issues/2956
# Here we will be patching the OpenAIWrapper to fix the issue.
# With this patch, you only need to register the client once with ADS.
# For example:
#
# from ads.llm.autogen.client_v02 import LangChainModelClient, register_custom_client
# register_custom_client(LangChainModelClient)
#
# This patch will auto-register the custom LLM to all new agents.
# So there is no need to call `register_model_client()` on each agent.
OpenAIWrapper._original_register_default_client = OpenAIWrapper._register_default_client


def _new_register_default_client(
    self: OpenAIWrapper, config: Dict[str, Any], openai_config: Dict[str, Any]
) -> None:
    """This is a patched version of the _register_default_client() method
    to automatically register custom client for agents.
    """
    model_client_cls_name = config.get("model_client_cls")
    if model_client_cls_name in custom_clients:
        self._clients.append(PlaceHolderClient(config))
        self.register_model_client(custom_clients[model_client_cls_name])
    else:
        self._original_register_default_client(
            config=config, openai_config=openai_config
        )


# Patch the _register_default_client() method
OpenAIWrapper._register_default_client = _new_register_default_client


def register_custom_client(client_class):
    """Registers custom client for AutoGen."""
    if client_class.__name__ not in custom_clients:
        custom_clients[client_class.__name__] = client_class


def _convert_to_langchain_tool(tool):
    """Converts the OpenAI tool spec to LangChain tool spec."""
    if tool["type"] == "function":
        tool = tool["function"]
        required = tool["parameters"].get("required", [])
        properties = copy.deepcopy(tool["parameters"]["properties"])
        for key in properties.keys():
            val = properties[key]
            val["default"] = key in required
        return {
            "title": tool["name"],
            "description": tool["description"],
            "properties": properties,
        }
    raise NotImplementedError(f"Type {tool['type']} is not supported.")


def _convert_to_openai_tool_call(tool_call):
    """Converts the LangChain tool call in AI message to OpenAI tool call."""
    return {
        "id": tool_call.get("id"),
        "function": {
            "name": tool_call.get("name"),
            "arguments": (
                ""
                if tool_call.get("args") is None
                else json.dumps(tool_call.get("args"))
            ),
        },
        "type": "function",
    }


class Message(AIMessage):
    """Represents message returned from the LLM."""

    @classmethod
    def from_message(cls, message: AIMessage):
        """Converts from LangChain AIMessage."""
        message = copy.deepcopy(message)
        message.__class__ = cls
        message.tool_calls = [
            _convert_to_openai_tool_call(tool) for tool in message.tool_calls
        ]
        return message

    @property
    def function_call(self):
        """Function calls."""
        return self.tool_calls


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LangChainModelClient(ModelClient):
    """Represents a model client wrapping a LangChain chat model."""

    def __init__(self, config: dict, **kwargs) -> None:
        super().__init__()
        logger.info("LangChain model client config: %s", str(config))
        # Make a copy of the config since we are popping some keys
        config = copy.deepcopy(config)
        # model_client_cls will always be LangChainModelClient
        self.client_class = config.pop("model_client_cls")

        # model_name is used in constructing the response.
        self.model_name = config.get("model", "")

        # If the config specified function_call_params,
        # Pop the params and use them only for tool calling.
        self.function_call_params = config.pop("function_call_params", {})

        # If the config specified invoke_params,
        # Pop the params and use them only for invoking.
        self.invoke_params = config.pop("invoke_params", {})

        # Import the LangChain class
        if "langchain_cls" not in config:
            raise ValueError("Missing langchain_cls in LangChain Model Client config.")
        self.langchain_cls = config.pop("langchain_cls")
        module_name, cls_name = str(self.langchain_cls).rsplit(".", 1)
        langchain_module = importlib.import_module(module_name)
        langchain_cls = getattr(langchain_module, cls_name)

        # If the config specified client_params,
        # Only use the client_params to initialize the LangChain model.
        # Otherwise, use the config
        self.client_params = config.get("client_params", config)

        # Initialize the LangChain client
        self.model = langchain_cls(**self.client_params)

    def create(self, params) -> ModelClient.ModelClientResponseProtocol:
        """Creates a LLM completion for a given config.

        Parameters
        ----------
        params : dict
            OpenAI API compatible parameters, including all the keys from llm_config.

        Returns
        -------
        ModelClientResponseProtocol
            Response from LLM

        """
        streaming = params.get("stream", False)
        # TODO: num_of_responses
        num_of_responses = params.get("n", 1)

        messages = copy.deepcopy(params.get("messages", []))

        # OCI Gen AI does not allow empty message.
        if str(self.langchain_cls).endswith("oci_generative_ai.ChatOCIGenAI"):
            for message in messages:
                if len(message.get("content", "")) == 0:
                    message["content"] = " "

        invoke_params = copy.deepcopy(self.invoke_params)

        tools = params.get("tools")
        if tools:
            model = self.model.bind_tools(
                [_convert_to_langchain_tool(tool) for tool in tools]
            )
            invoke_params.update(self.function_call_params)
        else:
            model = self.model

        response = SimpleNamespace()
        response.choices = []
        response.model = self.model_name
        response.usage = Usage()

        if streaming and messages:
            # If streaming is enabled and has messages, then iterate over the chunks of the response.
            raise NotImplementedError()
        else:
            # If streaming is not enabled, send a regular chat completion request
            ai_message = model.invoke(messages, **invoke_params)
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
        return [choice.message for choice in response.choices]

    def cost(self, response: ModelClient.ModelClientResponseProtocol) -> float:
        response.cost = 0
        return 0

    @staticmethod
    def get_usage(response: ModelClient.ModelClientResponseProtocol) -> Dict:
        """Return usage summary of the response using RESPONSE_USAGE_KEYS."""
        return asdict(response.usage)
