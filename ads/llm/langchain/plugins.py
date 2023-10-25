#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import oci
import subprocess
import functools
import requests
import logging

from typing import Any, Mapping, Dict, List, Optional

# TODO: Switch to runtime_dependency
# from ads.common.decorator.runtime_dependency import (
#     runtime_dependency,
#     OptionalDependency,
# )
# @runtime_dependency(module="langchain", install_from=OptionalDependency.LANGCHAIN)
try:
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.llms.base import LLM
    from langchain.pydantic_v1 import root_validator, Field, Extra
except ImportError as e:
    print("Pip install `langchain`")
    pass

try:
    from oci.generative_ai import GenerativeAiClient, models
except ImportError as e:
    print("Pip install `oci` with correct version")

logger = logging.getLogger(__name__)


class NotAuthorizedError(oci.exceptions.ServiceError):
    pass


# Move to constant.py
class TASK:
    TEXT_GENERATION = "text_generation"
    SUMMARY_TEXT = "summary_text"


class LengthParamOptions:
    SHORT = "SHORT"
    MEDIUM = "MEDIUM"
    LONG = "LONG"
    AUTO = "AUTO"


class FormatParamOptions:
    PARAGRAPH = "PARAGRAPH"
    BULLETS = "BULLETS"
    AUTO = "AUTO"


class ExtractivenessParamOptions:
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    AUTO = "AUTO"


class OCIGenerativeAIModelOptions:
    COHERE_COMMAND = "cohere.command"
    COHERE_COMMAND_LIGHT = "cohere.command-light"


DEFAULT_SERVICE_ENDPOINT = (
    "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"
)
DEFAULT_TIME_OUT = 300
DEFAULT_CONTENT_TYPE_JSON = "application/json"


def authenticate_with_security_token(profile):
    return subprocess.check_output(
        f"oci session authenticate --profile-name {profile} --region us-ashburn-1 --tenancy-name bmc_operator_access",
        shell=True,
    )


def with_oci_token(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            results = func(self, *args, **kwargs)
            return results
        except (
            oci.exceptions.ServiceError,
            oci.exceptions.ProfileNotFound,
            NotAuthorizedError,
        ) as ex:
            if self.config_profile and ex.status in [401, 404]:
                authenticate_with_security_token(self.config_profile)
            _auth = get_oci_auth(profile=self.config_profile)
            if hasattr(self, "client"):
                _client_kwargs = self.client_kwargs or {}
                self.client = GenerativeAiClient(**_auth, **_client_kwargs)
            if hasattr(self, "signer"):
                self.signer = _auth.get("signer")
            return func(self, *args, **kwargs)

    return wrapper


def get_oci_auth(
    config_location=oci.config.DEFAULT_LOCATION,
    profile=oci.config.DEFAULT_PROFILE,
):
    oci_config = oci.config.from_file(
        file_location=config_location, profile_name=profile
    )
    if "security_token_file" in oci_config and "key_file" in oci_config:
        token_file = oci_config["security_token_file"]
        with open(token_file, "r", encoding="utf-8") as f:
            token = f.read()
        private_key = oci.signer.load_private_key_from_file(oci_config["key_file"])
        signer = oci.auth.signers.SecurityTokenSigner(token, private_key)
        oci_auth = {"config": oci_config, "signer": signer}
    else:
        oci_auth = {"config": oci_config}
    return oci_auth


class OCILLM(LLM):
    """Base OCI LLM class. Contains common attributes."""

    config: dict = None
    config_profile: str = None
    config_location: str = oci.config.DEFAULT_LOCATION
    signer: Any = None

    max_tokens: int = 256
    """Denotes the number of tokens to predict per generation."""

    temperature: float = 0.1
    """A non-negative float that tunes the degree of randomness in generation."""

    k: int = 0
    """Number of most likely tokens to consider at each step."""

    p: int = 0.9
    """Total probability mass of tokens to consider at each step."""

    stop: Optional[List[str]] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def _try_init_oci_auth(cls, values: Dict) -> dict:
        """Returns {"config": "", "signer": ""}."""
        allowed_params = ["config", "config_profile", "signer", "config_location"]
        params = {k: v for k, v in values.items() if k in allowed_params}
        _config_location = params.get("config_location", oci.config.DEFAULT_LOCATION)
        if params.get("config_profile"):
            return get_oci_auth(
                profile=params["config_profile"], config_location=_config_location
            )
        return params

    @classmethod
    def _create_default_signer():
        """Use ads to help to create oci.signer.Signer object."""
        # TODO
        # ads.set_auth(config=other_config)
        # return default_signer()["signer"]
        raise NotImplementedError


class GenerativeAI(OCILLM):
    """GenerativeAI

    To use, you should have the ``oci`` python package installed

    Example
    -------

    .. code-block:: python

        from ads.llm import GenerativeAI
        gen_ai = GenerativeAI(*args, **kwargs) # some params config
        gen_ai("Tell me a joke.")

    """

    client: Any  #: :meta private:

    model: Optional[
        str
    ] = OCIGenerativeAIModelOptions.COHERE_COMMAND  # "cohere.command"
    """Model name to use."""

    # For text generation task.
    frequency_penalty: float = None
    """Penalizes repeated tokens according to frequency. Between 0 and 1."""

    presence_penalty: float = None
    """Penalizes repeated tokens. Between 0 and 1."""

    truncate: Optional[str] = None
    """Specify how the client handles inputs longer than the maximum token
    length: Truncate from START, END or NONE"""

    # For summarization task.
    length: str = LengthParamOptions.AUTO

    format: str = FormatParamOptions.PARAGRAPH

    extractiveness: str = ExtractivenessParamOptions.AUTO

    additional_command: str = ""

    endpoint_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Optional attributes passed to the generate_text/summarize_text function."""

    client_kwargs: Dict[str, Any] = {"service_endpoint": DEFAULT_SERVICE_ENDPOINT}
    """Holds any client parametes for creating GenerativeAiClient"""

    compartment_id: str = None

    task: str = TASK.TEXT_GENERATION

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""
        if values.get("client") is not None:
            return values

        _auth = cls._try_init_oci_auth(values)
        _client_kwargs = values["client_kwargs"] or {}
        try:
            import oci

            values["client"] = GenerativeAiClient(**_auth, **_client_kwargs)
        except ImportError:
            raise ImportError(
                "Could not import oci python package. "
                "Please install it with `pip install oci`."
            )
        return values

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            **{
                "model": self.model,
                "task": self.task,
                "client_kwargs": self.client_kwargs,
                "endpoint_kwargs": self.endpoint_kwargs,
            },
            **self._default_params,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "OCIGenerativeAI"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OCIGenerativeAI API."""
        # properties of oci.generative_ai.models.GenerateTextDetails
        return (
            {
                "compartment_id": self.compartment_id,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_k": self.k,
                "top_p": self.p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "truncate": self.truncate,
                "serving_mode": models.OnDemandServingMode(model_id=self.model),
            }
            if self.task == "text_generation"
            else {
                "serving_mode": models.OnDemandServingMode(model_id=self.model),
                "compartment_id": self.compartment_id,
                "temperature": self.temperature,
                "length": self.length,
                "format": self.format,
                "extractiveness": self.extractiveness,
                "additional_command": self.additional_command,
            }
        )

    def _invocation_params(self, stop: Optional[List[str]], **kwargs: Any) -> dict:
        params = self._default_params
        if self.task == TASK.SUMMARY_TEXT:
            return {**params}

        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            params["stop_sequences"] = self.stop
        else:
            params["stop_sequences"] = stop
        return {**params, **kwargs}

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        """Call out to OCIGenerativeAI's generate endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = gen_ai("Tell me a joke.")
        """

        params = self._invocation_params(stop, **kwargs)

        try:
            response = (
                self.completion_with_retry(prompts=[prompt], **params)
                if self.task == TASK.TEXT_GENERATION
                else self.completion_with_retry(input=prompt, **params)
            )
        except Exception as ex:
            logger.error(
                "Error occur when invoking oci service api."
                f"DEBUG INTO: task={self.task}, params={params}, prompt={prompt}"
            )
            raise

        return self._process_response(response, params.get("num_generations", 1))

    def _process_response(self, response: Any, num_generations: int = 1) -> str:
        if self.task == TASK.SUMMARY_TEXT:
            return response.data.summary

        return (
            response.data.generated_texts[0][0].text
            if num_generations == 1
            else [gen.text for gen in response.data.generated_texts[0]]
        )

    @with_oci_token
    def completion_with_retry(self, **kwargs: Any) -> Any:
        _model_kwargs = {**kwargs}
        _endpoint_kwargs = self.endpoint_kwargs or {}

        if self.task == TASK.TEXT_GENERATION:
            return self.client.generate_text(
                models.GenerateTextDetails(**_model_kwargs), **_endpoint_kwargs
            )
        elif self.task == TASK.SUMMARY_TEXT:
            return self.client.summarize_text(
                models.SummarizeTextDetails(**_model_kwargs), **_endpoint_kwargs
            )
        else:
            raise ValueError("Unsupported tasks.")

    def batch_completion(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        num_generations: int = 1,
        **kwargs: Any,
    ):
        if self.task == TASK.SUMMARY_TEXT:
            raise NotImplementedError(
                f"task={TASK.SUMMARY_TEXT} does not support batch_completion. "
            )

        return self._call(
            prompt=prompt,
            stop=stop,
            run_manager=run_manager,
            num_generations=num_generations,
            **kwargs,
        )


class OCIModelDeployment(OCILLM):
    best_of = 1
    endpoint: str = None

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"endpoint": self.endpoint},
            **self._default_params,
        }

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Dont do anything if client provided externally"""
        if values.get("signer") is not None:
            return values

        # _signer = cls.create_default_signer(values.get("config"))
        try:
            import requests

            # values["signer"] = _signer
        except ImportError:
            raise ImportError(
                "Could not import requests python package. "
                "Please install it with `pip install requests`."
            )
        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to OCI Data Science Model Deployment TGI endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = gen_ai("Tell me a joke.")
        """
        params = self._invocation_params(stop, **kwargs)
        body = self._construct_json_body(prompt, params)
        response = self.send_request(
            data=body, endpoint=self.endpoint, timeout=DEFAULT_TIME_OUT
        )

        return str(response.get("generated_text", response))

    @with_oci_token
    def send_request(
        self,
        data,
        endpoint: str,
        header: dict = {},
        **kwargs,
    ):
        header["Content-Type"] = (
            header.pop("content_type", DEFAULT_CONTENT_TYPE_JSON)
            or DEFAULT_CONTENT_TYPE_JSON
        )
        request_kwargs = {"json": data}
        request_kwargs["headers"] = header
        request_kwargs["auth"] = self.signer

        try:
            response = requests.post(endpoint, **request_kwargs, **kwargs)
            response_json = response.json()
            if response_json.get("code") == "NotAuthorizedOrNotFound":
                response_json["status"] = response.status_code
                response_json["headers"] = response.headers
                raise NotAuthorizedError(**response_json)

            if response.status_code != 200:
                raise ValueError(
                    f"Failed to invoke endpoint. response status code={response.status_code}"
                )
        except NotAuthorizedError:
            raise
        except Exception:
            logger.error(
                f"DEBUG INFO: request_kwargs={request_kwargs},"
                f"status_code={response.status_code}, "
                f"content={response._content}"
            )
            raise

        return response_json

    def _construct_json_body(self, prompt, params):
        raise NotImplementedError


class OCIModelDeploymentTGI(OCIModelDeployment):
    """_summary_

    Args:
        OCIModelDeployment (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_

    Example
    -------
    >>> # Initialize the LLM client
    >>> oci_md = OCIModelDeploymentTGI(*args, **kwargs)
    >>> # Get a response given a prompt
    >>> oci_md("Tell me a joke.")

    """

    do_sample = True
    watermark = True
    return_full_text = True

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "oci_model_deployment_tgi_endpoint"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for invoking OCI model deployment TGI endpoint."""
        return {
            "best_of": self.best_of,
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.k
            if self.k > 0
            else None,  # `top_k` must be strictly positive'
            "top_p": self.p,
            "do_sample": self.do_sample,
            "return_full_text": self.return_full_text,
            "watermark": self.watermark,
        }

    def _invocation_params(self, stop: Optional[List[str]], **kwargs: Any) -> dict:
        params = self._default_params
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            params["stop"] = self.stop
        elif stop is not None:
            params["stop"] = stop
        else:  # don't set stop in param as None. TGI not accept stop=null.
            pass
        return {**params, **kwargs}

    def _construct_json_body(self, prompt, params):
        return {
            "inputs": prompt,
            "parameters": params,
        }


# class OCIModelDeploymentvLLM(OCIModelDeployment):
#     """Not finished yet. Need to clarify the usage first."""

#     n: int = 1
#     """Number of output sequences to return for the given prompt."""

#     presence_penalty: float = 0.0
#     """Float that penalizes new tokens based on whether they appear in the
#     generated text so far"""

#     frequency_penalty: float = 0.0
#     """Float that penalizes new tokens based on their frequency in the
#     generated text so far"""

#     use_beam_search: bool = False
#     """Whether to use beam search instead of sampling."""

#     ignore_eos: bool = False
#     """Whether to ignore the EOS token and continue generating tokens after
#     the EOS token is generated."""

#     logprobs: Optional[int] = None
#     """Number of log probabilities to return per output token."""

#     @property
#     def _llm_type(self) -> str:
#         """Return type of llm."""
#         return "oci_model_deployment_vllm_endpoint"

#     @property
#     def _default_params(self) -> Dict[str, Any]:
#         """Get the default parameters for invoking OCI model deployment vllm endpoint."""
#         return {
#             "n": self.n,
#             "best_of": self.best_of,
#             "max_tokens": self.max_tokens,
#             "top_k": self.k,
#             "top_p": self.p,
#             "temperature": self.temperature,
#             "presence_penalty": self.presence_penalty,
#             "frequency_penalty": self.frequency_penalty,
#             "ignore_eos": self.ignore_eos,
#             "use_beam_search": self.use_beam_search,
#             "logprobs": self.logprobs,
#         }

#     def _invocation_params(self, stop: Optional[List[str]], **kwargs: Any) -> dict:
#         params = self._default_params
#         if self.stop is not None and stop is not None:
#             raise ValueError("`stop` found in both the input and default params.")
#         elif self.stop is not None:
#             params["stop"] = self.stop
#         else:
#             params["stop"] = stop
#         return {**params, **kwargs}
