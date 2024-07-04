#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import re
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
from packaging import version

logger = logging.getLogger(__name__)

OCI_LLM_ENDPOINT = "OCI_LLM_ENDPOINT"
MIN_ADS_VERSION = "2.9.1"  # "2.11.13"


class UnsupportedAdsVersionError(Exception):
    def __init__(self, current_version: str, required_version: str):
        super().__init__(
            f"The `ads` version {current_version} currently installed is incompatible with "
            "the `langchain` version in use. To resolve this issue, please upgrade to `ads` "
            f"version {required_version} or later using the command: `pip install oracle-ads -U`"
        )


class InferenceBackend(Protocol):
    """Protocol for the inference backend."""

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompts."""

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        ...


def _validate_dependency(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to validate the presence and version of the `ads` package.

    Raises:
        ImportError: If the `ads` package is not installed.
        UnsupportedAdsVersionError: If the installed `ads` version is lower than the required version.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            from ads import __version__ as ads_version

            if version.parse(ads_version) < version.parse(MIN_ADS_VERSION):
                raise UnsupportedAdsVersionError(ads_version, MIN_ADS_VERSION)

        except ImportError as ex:
            raise ImportError(
                "Could not import `ads` python package. "
                "Please install it with `pip install oracle-ads`."
            ) from ex
        return func(*args, **kwargs)

    return wrapper


def _is_hex_string(data: str) -> bool:
    """
    Check if the provided string is a valid hexadecimal string.

    Args:
        data (str): The string to check.

    Returns:
        bool: True if the string is a valid hexadecimal string, False otherwise.
    """
    if not isinstance(data, str):
        return False
    hex_pattern = r"^[0-9a-fA-F]+$"
    return bool(re.match(hex_pattern, data))


def _deserialize_function_from_hex(
    hex_data: str, allow_unsafe_deserialization: Optional[bool] = False
) -> Callable[..., Any]:
    """
    Deserialize a pickled function from a hexadecimal string.

    Args:
        hex_data (str): The hexadecimal string to deserialize.
        allow_unsafe_deserialization (Optional[bool]): Whether to allow unsafe deserialization.

    Returns:
        Callable[..., Any]: The deserialized function.

    Raises:
        ValueError: If deserialization is not allowed or fails.
        ImportError: If the `cloudpickle` package is not installed.
    """
    if not allow_unsafe_deserialization:
        raise ValueError(
            "Deserialization requires opting in to allow unsafe deserialization. "
            "Set allow_unsafe_deserialization=True to proceed. "
            "Be aware that deserializing untrusted data can execute arbitrary code."
        )

    try:
        import cloudpickle
    except ImportError as e:
        raise ImportError("Please install cloudpickle>=2.0.0. Error: {e}")

    try:
        return cloudpickle.loads(bytes.fromhex(hex_data))
    except Exception as e:
        raise ValueError(f"Failed to deserialize function from hex string. Error: {e}")


class OCIModelDeployment(BaseLLM):
    """LLM deployed on OCI Data Science Model Deployment

    To use, you must provide the model HTTP endpoint from your deployed
    model, e.g. https://<MD_OCID>/predict.

    To authenticate, `oracle-ads` has been used to automatically load
    credentials: https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html

    Make sure to have the required policies to access the OCI Data
    Science Model Deployment endpoint. See:
    https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-policies-auth.htm#model_dep_policies_auth__predict-endpoint

    Example:
        .. code-block:: python

            from langchain_community.llms import OCIModelDeployment

            oci_md = OCIModelDeployment(
                endpoint="https://<MD_OCID>/predict",
                model="mymodel"
            )
    """

    auth: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    """
    The authentication dictionary used for OCI API requests. Default is an empty dictionary.
    If not provided, it will be autogenerated based on the environment variables.
    For more details, refer to:
    https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html.
    """

    endpoint: Optional[str] = None
    """The URI of the endpoint from the deployed model."""

    max_tokens: Optional[int] = 256
    """Denotes the number of tokens to predict per generation."""

    temperature: Optional[float] = 0.2
    """A non-negative float that tunes the degree of randomness in generation."""

    k: Optional[int] = 50
    """Number of most likely tokens to consider at each step."""

    p: Optional[float] = 0.75
    """Total probability mass of tokens to consider at each step."""

    best_of: Optional[int] = 1
    """Generates best_of completions server-side and returns the "best"
    (the one with the highest log probability per token).
    """

    stop: Optional[List[str]] = None
    """Stop words to use when generating. Model output is cut off
    at the first occurrence of any of these substrings."""

    model_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict)
    """
    Additional keyword arguments for the model. The default values will be taken based on the
    selected framework. Use this variable to pass model sampling parameters.
    Use `OCIModelDeployment.help("vllm")` to get details about the particular framework.
    """

    allow_unsafe_deserialization: bool = False
    """
    Flag to allow unsafe deserialization of pickled functions. Default is False.
    This will be used to deserialize `transform_input_fn` and `transform_output_fn` if provided.

    Important: Data can be compromised by a malicious actor if not handled properly, including
    a malicious payload that, when deserialized with pickle, can execute arbitrary code on your machine.
    """

    transform_input_fn: Optional[Union[str, Callable[..., Dict]]] = None
    """
    Function to convert `{prompt, stop, **model_kwargs}` into a JSON-compatible request object that is accepted by the endpoint.
    By default, the one implemented at the framework level will be used. However, if the
    default behavior needs to be changed, this function can be used.

    Important: This method will de serialized with cloudpickle.
    Data can be compromised by a malicious actor if not handled properly, including
    a malicious payload that, when deserialized with pickle, can execute arbitrary code on your machine.
    """

    transform_output_fn: Optional[Union[str, Callable[..., List[Generation]]]] = None
    """
    Function to transform the response from the endpoint to the `List[Generation]` before returning it.
    By default, the one implemented at the framework level will be used. However, if the
    default behavior needs to be changed, this function can be used.

    Important: This method will de serialized with cloudpickle.
    Data can be compromised by a malicious actor if not handled properly, including
    a malicious payload that, when deserialized with pickle, can execute arbitrary code on your machine.
    """

    inference_framework: Optional[str] = "generic"
    """
    The framework used for inference. Examples include 'vllm', 'tgi', 'generic', 'llama.cpp'.
    Use `OCIModelDeployment.supported_frameworks()` to see the list of supported frameworks.
    The `generic` is used by default.
    """

    _backend: InferenceBackend = Field(default=None, exclude=True)
    """
    The backend interface for the inference model.
    It will be created based on the selected inference framework.
    """

    class Config:
        extra = Extra.forbid
        # underscore_attrs_are_private = True

    @staticmethod
    @_validate_dependency
    def supported_frameworks() -> List[str]:
        """
        Get the list of supported inference frameworks.

        Returns:
            List[str]: The supported inference frameworks.
        """
        from ads.llm.langchain.inference_backend import InferenceBackendFactory

        return InferenceBackendFactory.supported_frameworks()

    @classmethod
    def _extract_child_class_attributes(cls):
        base_class_attributes = [
            "max_tokens",
            "temperature",
            "k",
            "p",
            "best_of",
            "stop",
        ]

        if not cls.__bases__ or cls.__name__ == "OCIModelDeployment":
            return base_class_attributes

        parent_cls = cls.__bases__[0]
        child_attributes = set(cls.__annotations__.keys())
        parent_attributes = set(parent_cls.__annotations__.keys())
        all_attributes = list(
            child_attributes - parent_attributes | set(base_class_attributes)
        )
        return all_attributes

    @root_validator()
    @_validate_dependency
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and set up the environment.

        Args:
            values (Dict[str, Any]): The input values.

        Returns:
            Dict[str, Any]: The validated and possibly modified values.
        """
        from ads.common.auth import default_signer
        from ads.llm.langchain.inference_backend import InferenceBackendFactory

        values["auth"] = values.get("auth") or default_signer()
        values["endpoint"] = get_from_dict_or_env(values, "endpoint", OCI_LLM_ENDPOINT)

        if values.get("transform_input_fn") and _is_hex_string(
            values["transform_input_fn"]
        ):
            values["transform_input_fn"] = _deserialize_function_from_hex(
                hex_data=values["transform_input_fn"],
                allow_unsafe_deserialization=values.get("allow_unsafe_deserialization"),
            )
        if values.get("transform_output_fn") and _is_hex_string(
            values["transform_output_fn"]
        ):
            values["transform_output_fn"] = _deserialize_function_from_hex(
                hex_data=values["transform_output_fn"],
                allow_unsafe_deserialization=values.get("allow_unsafe_deserialization"),
            )

        framework_kwargs = {
            arg: values.get(arg) for arg in cls._extract_child_class_attributes()
        }

        # setup backend
        values["_backend"] = InferenceBackendFactory.get_backend(
            values.get("inference_framework")
        )(**{**values, "framework_kwargs": framework_kwargs})

        return values

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """
        Get the identifying parameters of the backend.

        Returns:
            Dict[str, Any]: The identifying parameters.
        """

        return {
            "inference_framework": self.inference_framework,
            **self._backend._identifying_params,
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
        return self._backend._generate(
            prompts=prompts, stop=stop, run_manager=run_manager, **kwargs
        )

    @property
    def _llm_type(self) -> str:
        return "oci_model_deployment"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True


class OCIModelDeploymentVLLM(OCIModelDeployment):
    """VLLM deployed on OCI Data Science Model Deployment

    To use, you must provide the model HTTP endpoint from your deployed
    model, e.g. https://<MD_OCID>/predict.

    To authenticate, `oracle-ads` has been used to automatically load
    credentials: https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html

    Make sure to have the required policies to access the OCI Data
    Science Model Deployment endpoint. See:
    https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-policies-auth.htm#model_dep_policies_auth__predict-endpoint

    Example:
        .. code-block:: python

            from langchain_community.llms import OCIModelDeploymentVLLM

            oci_md = OCIModelDeploymentVLLM(
                endpoint="https://<MD_OCID>/predict",
                model="mymodel"
            )

    """

    model: str = "odsc-llm"
    """The name of the model."""

    n: int = 1
    """Number of output sequences to return for the given prompt."""

    k: int = -1
    """Number of most likely tokens to consider at each step."""

    frequency_penalty: float = 0.0
    """Penalizes repeated tokens according to frequency. Between 0 and 1."""

    presence_penalty: float = 0.0
    """Penalizes repeated tokens. Between 0 and 1."""

    use_beam_search: bool = False
    """Whether to use beam search instead of sampling."""

    ignore_eos: bool = False
    """Whether to ignore the EOS token and continue generating tokens after
    the EOS token is generated."""

    logprobs: Optional[int] = None
    """Number of log probabilities to return per output token."""

    inference_framework: Optional[str] = "vllm"

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "oci_model_deployment_vllm_endpoint"


class OCIModelDeploymentTGI(OCIModelDeployment):
    """OCI Data Science Model Deployment TGI Endpoint.

    To use, you must provide the model HTTP endpoint from your deployed
    model, e.g. https://<MD_OCID>/predict.

    To authenticate, `oracle-ads` has been used to automatically load
    credentials: https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html

    Make sure to have the required policies to access the OCI Data
    Science Model Deployment endpoint. See:
    https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-policies-auth.htm#model_dep_policies_auth__predict-endpoint

    Example:
        .. code-block:: python

            from langchain_community.llms import ModelDeploymentTGI

            oci_md = ModelDeploymentTGI(endpoint="https://<MD_OCID>/predict")
    """

    do_sample: bool = True
    """If set to True, this parameter enables decoding strategies such as
    multi-nominal sampling, beam-search multi-nominal sampling, Top-K
    sampling and Top-p sampling.
    """

    watermark: bool = True
    """Watermarking with `A Watermark for Large Language Models <https://arxiv.org/abs/2301.10226>`_.
    Defaults to True."""

    return_full_text = False
    """Whether to prepend the prompt to the generated text. Defaults to False."""

    inference_framework: Optional[str] = "tgi"

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "oci_model_deployment_tgi_endpoint"
