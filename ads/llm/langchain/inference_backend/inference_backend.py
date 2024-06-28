#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from typing import Any, Callable, Dict, List, Optional, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.pydantic_v1 import BaseModel, Extra, Field

from ads.llm.langchain.inference_backend.model_invoker import ModelInvoker
from ads.llm.langchain.inference_backend.utils import serialize_function_to_hex

logger = logging.getLogger(__name__)


class InferenceBackend(BaseModel):
    """
    Base class for inference backends, providing common functionality for
    handling model invocations and transformations.
    """

    auth: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    endpoint: Optional[str] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    transform_input_fn: Optional[Callable[..., Dict[str, Any]]] = None
    transform_output_fn: Optional[Callable[..., str]] = None

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

    def _process_response(self, response_json: Dict[str, Any]) -> str:
        """
        Processes the response from the model.

        Parameters
        ----------
        response_json (Dict[str, Any]): The response JSON.

        Returns
        -------
        str: The processed response.
        """
        raise NotImplementedError

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Return default parameters for the model."""
        raise NotImplementedError

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters for the model."""
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
            **self._default_params,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Invokes the model with the given prompt and additional parameters.

        Parameters
        ----------
        prompt (str): The prompt for the model.
        stop (Optional[List[str]]): Optional list of stop words.
        run_manager (Optional[CallbackManagerForLLMRun]): Optional callback manager.
        **kwargs (Any): Additional keyword arguments.

        Returns
        -------
        str: The model's response.
        """

        request_kwargs = kwargs.pop("request_kwargs", {}) or {}
        model_invoker = ModelInvoker(
            endpoint=self.endpoint, auth=self.auth, **request_kwargs
        )

        stop = stop or self._default_params.get("stop", []) or []

        transform_input_fn = self.transform_input_fn or self._construct_json_body
        transform_output_fn = self.transform_output_fn or self._process_response

        return transform_output_fn(
            model_invoker.invoke(
                params=transform_input_fn(
                    prompt=prompt, stop=stop, params=self._default_params
                )
            )
        )


class InferenceBackendGeneric(InferenceBackend):
    """
    A generic implementation of the InferenceBackend class.
    """

    class ModelParams(BaseModel):
        """
        Represents the default model parameters.
        """

        model: Optional[str] = "odsc-llm"
        temperature: Optional[float] = 0.7
        top_p: Optional[float] = 0.9
        n: Optional[int] = 1
        max_tokens: Optional[int] = 500
        seed: Optional[int] = None
        stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
        top_k: Optional[int] = 50
        best_of: Optional[int] = None

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

    def _process_response(self, response_json: Dict[str, Any]) -> str:
        """
        Processes the response from the model.

        Parameters
        ----------
        response_json (Dict[str, Any]): The response JSON.

        Returns
        -------
        str: The processed response.
        """
        if response_json["object"] == "error":
            return response_json.get("message", "")
        return response_json["choices"][0]["text"]

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Return default parameters for the model."""
        return self.ModelParams(**self.model_kwargs).dict()


class InferenceBackendVLLM(InferenceBackendGeneric):
    """
    An implementation of the InferenceBackend class for the Versatile Large Language Model (vLLM).
    """

    ...


class InferenceBackendTGI(InferenceBackendGeneric):
    """
    An implementation of the InferenceBackend class for the TGI (Text Generation Inference) model.
    """

    ...


class InferenceBackendLLamaCPP(InferenceBackendGeneric):
    """
    An implementation of the InferenceBackend class for the Llama.CPP model.
    """

    ...
