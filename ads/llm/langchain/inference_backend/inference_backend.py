#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from typing import Any, Callable, Dict, List, Optional, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import Generation, LLMResult
from langchain_core.pydantic_v1 import BaseModel, Extra, Field

from ads.llm.langchain.inference_backend.const import InferenceFramework
from ads.llm.langchain.inference_backend.model_invoker import ModelInvoker
from ads.llm.langchain.inference_backend.utils import serialize_function_to_hex
from ads.model import framework

logger = logging.getLogger(__name__)


class InferenceBackend(BaseModel):
    """
    Base class for inference backends, providing common functionality for
    handling model invocations and transformations.
    """

    auth: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    endpoint: Optional[str] = None
    framework_kwargs: Dict[str, Any] = Field(default_factory=dict)
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    transform_input_fn: Optional[Callable[..., Dict[str, Any]]] = None
    transform_output_fn: Optional[Callable[..., str]] = None
    allow_unsafe_deserialization: bool = False

    def _construct_json_body(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Constructs the request body as a dictionary (JSON).

        Parameters
        ----------
        prompt (str): The prompt for the model.
        stop (Optional[List[str]]): Optional list of stop words.
        params (Optional[Dict[str, Any]]): Additional parameters.

        Returns
        -------
        Dict[str, Any]: The constructed JSON body.
        """
        raise NotImplementedError

    def _process_response(self, response_json: Dict[str, Any]) -> List[Generation]:
        """
        Processes the response from the model.

        Parameters
        ----------
        response_json (Dict[str, Any]): The response JSON.

        Returns
        -------
        List[Generation]: The processed response.
        """
        raise NotImplementedError

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Return default parameters for the model."""
        framework_kwargs = {**self.framework_kwargs}
        return self.ModelParams(
            **{
                **self.model_kwargs,
                **{
                    "top_k": framework_kwargs.pop("k", None),
                    "top_p": framework_kwargs.pop("p", None),
                    **framework_kwargs,
                },
            }
        ).dict()

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters for the model."""
        default_params = {**self._default_params} or {}
        default_params.update(
            {
                "k": default_params.pop("top_k", None),
                "p": default_params.pop("top_p", None),
            }
        )
        framework_kwargs = {
            attr: default_params.pop(attr, None)
            for attr in self.framework_kwargs.keys()
        }

        return {
            "endpoint": self.endpoint,
            "transform_input_fn": (
                None
                if self.transform_input_fn is None
                else serialize_function_to_hex(self.transform_input_fn)
            ),
            "transform_output_fn": (
                None
                if self.transform_output_fn is None
                else serialize_function_to_hex(self.transform_output_fn)
            ),
            "model_kwargs": default_params,
            **framework_kwargs,
            "allow_unsafe_deserialization": self.allow_unsafe_deserialization,
        }

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """
        Run the LLM on the given prompts and input.

        Args:
            prompts (List[str]): The list of prompts to process.
            stop (Optional[List[str]]): Optional stop words.
            run_manager (Optional[CallbackManagerForLLMRun]): Optional run manager.
            kwargs (Any): Additional keyword arguments.

        Returns:
            LLMResult: The result from the backend.
        """

        generations: List[List[Generation]] = []

        request_kwargs = kwargs.pop("request_kwargs", {}) or {}
        headers = kwargs.pop("headers", {}) or {}
        model_invoker = ModelInvoker(
            endpoint=self.endpoint, auth=self.auth, **request_kwargs
        )
        stop = stop or self._default_params.get("stop", []) or []
        transform_input_fn = self.transform_input_fn or self._construct_json_body
        transform_output_fn = self.transform_output_fn or self._process_response

        for prompt in prompts:
            prompt_result: List[Generation] = transform_output_fn(
                model_invoker.invoke(
                    params=transform_input_fn(
                        prompt=prompt, stop=stop, params=self._default_params
                    ),
                    headers=headers,
                )
            )
            generations.append(prompt_result)

        return LLMResult(generations=generations)


class InferenceBackendGeneric(InferenceBackend):
    """
    A generic implementation of the InferenceBackend class.
    """

    class ModelParams(BaseModel):
        """
        Represents the default model parameters.
        """

        model: Optional[str] = "odsc-llm"
        temperature: Optional[float] = 0.2
        top_p: Optional[float] = 0.75
        n: Optional[int] = 1
        max_tokens: Optional[int] = 256
        seed: Optional[int] = None
        stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
        top_k: Optional[int] = 50
        best_of: Optional[int] = 1

        class Config:
            extra = Extra.allow

    def _construct_json_body(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Constructs the request body as a dictionary (JSON).

        Parameters
        ----------
        prompt (str): The prompt for the model.
        stop (Optional[List[str]]): Optional list of stop words.
        params (Optional[Dict[str, Any]]): Additional parameters.

        Returns
        -------
        Dict[str, Any]: The constructed JSON body.
        """
        return {**(params or {}), "prompt": prompt, "stop": stop}

    def _process_response(self, response_json: Dict[str, Any]) -> List[Generation]:
        """
        Processes the response from the model.

        Parameters
        ----------
        response_json (Dict[str, Any]): The response JSON.

        Returns
        -------
        List[Generation]: The processed response.
        """

        # if response_json["object"] == "error":
        #   return response_json.get("message", "")

        return [
            Generation(
                text=choice.get("text", ""),
                generation_info={
                    "finish_reason": choice.get("finish_reason"),
                    "logprobs": choice.get("logprobs"),
                    "index": choice.get("index"),
                },
            )
            for choice in response_json.get("choices", [])
        ]


class InferenceBackendVLLM(InferenceBackendGeneric):
    """
    An implementation of the InferenceBackend class for the vLLM.
    """


class InferenceBackendTGI(InferenceBackendGeneric):
    """
    An implementation of the InferenceBackend class for the TGI (Text Generation Inference) model.
    """


class InferenceBackendLLamaCPP(InferenceBackendGeneric):
    """
    An implementation of the InferenceBackend class for the Llama.CPP model.
    """
