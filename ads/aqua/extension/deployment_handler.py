#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import List, Union
from urllib.parse import urlparse

from tornado.web import HTTPError

from ads.aqua import logger
from ads.aqua.client.client import Client, ExtendedRequestError
from ads.aqua.client.openai_client import OpenAI
from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.common.enums import PredictEndpoints
from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.extension.errors import Errors
from ads.aqua.modeldeployment import AquaDeploymentApp
from ads.config import COMPARTMENT_OCID


class AquaDeploymentHandler(AquaAPIhandler):
    """
    Handler for Aqua Deployment REST APIs.

    Methods
    -------
    get(self, id: Union[str, List[str]])
        Retrieves a list of AQUA deployments or model info or logs by ID.
    post(self, *args, **kwargs)
        Creates a new AQUA deployment.
    read(self, id: str)
        Reads the AQUA deployment information.
    list(self)
        Lists all the AQUA deployments.
    get_deployment_config(self, model_id)
        Gets the deployment config for Aqua model.
    list_shapes(self)
        Lists the valid model deployment shapes.

    Raises
    ------
    HTTPError: For various failure scenarios such as invalid input format, missing data, etc.
    """

    @handle_exceptions
    def get(self, id: Union[str, List[str]] = None):
        """Handle GET request."""
        url_parse = urlparse(self.request.path)
        paths = url_parse.path.strip("/")
        if paths.startswith("aqua/deployments/config"):
            if not id or not isinstance(id, str):
                raise HTTPError(
                    400,
                    f"Invalid request format for {self.request.path}. "
                    "Expected a single model ID or a comma-separated list of model IDs.",
                )
            id = id.replace(" ", "")
            return self.get_deployment_config(
                model_id=id.split(",") if "," in id else id
            )
        elif paths.startswith("aqua/deployments/recommend_shapes"):
            if not id or not isinstance(id, str):
                raise HTTPError(
                    400,
                    f"Invalid request format for {self.request.path}. "
                    "Expected a single model OCID specified as --model_id",
                )
            id = id.replace(" ", "")
            return self.get_recommend_shape(model_id=id)
        elif paths.startswith("aqua/deployments/shapes"):
            return self.list_shapes()
        elif paths.startswith("aqua/deployments"):
            if not id:
                return self.list()
            return self.read(id)
        else:
            raise HTTPError(400, f"The request {self.request.path} is invalid.")

    @handle_exceptions
    def delete(self, model_deployment_id):
        return self.finish(AquaDeploymentApp().delete(model_deployment_id))

    @handle_exceptions
    def put(self, *args, **kwargs):  # noqa: ARG002
        """
        Handles put request for the activating and deactivating OCI datascience model deployments
        Raises
        ------
        HTTPError
            Raises HTTPError if inputs are missing or are invalid
        """
        url_parse = urlparse(self.request.path)
        paths = url_parse.path.strip("/").split("/")
        if len(paths) != 4 or paths[0] != "aqua" or paths[1] != "deployments":
            raise HTTPError(400, f"The request {self.request.path} is invalid.")

        model_deployment_id = paths[2]
        action = paths[3]
        if action == "activate":
            return self.finish(AquaDeploymentApp().activate(model_deployment_id))
        elif action == "deactivate":
            return self.finish(AquaDeploymentApp().deactivate(model_deployment_id))
        else:
            raise HTTPError(400, f"The request {self.request.path} is invalid.")

    @handle_exceptions
    def post(self, *args, **kwargs):  # noqa: ARG002
        """
        Handles post request for the deployment APIs
        Raises
        ------
        HTTPError
            Raises HTTPError if inputs are missing or are invalid
        """
        try:
            input_data = self.get_json_body()
        except Exception as ex:
            raise HTTPError(400, Errors.INVALID_INPUT_DATA_FORMAT) from ex

        if not input_data:
            raise HTTPError(400, Errors.NO_INPUT_DATA)

        model_deployment_id = input_data.pop("model_deployment_id", None)
        if model_deployment_id:
            self.finish(
                AquaDeploymentApp().update(
                    model_deployment_id=model_deployment_id, **input_data
                )
            )
        else:
            self.finish(AquaDeploymentApp().create(**input_data))

    def read(self, id):
        """Read the information of an Aqua model deployment."""
        return self.finish(AquaDeploymentApp().get(model_deployment_id=id))

    def list(self):
        """List Aqua models."""
        # If default is not specified,
        # jupyterlab will raise 400 error when argument is not provided by the HTTP request.
        compartment_id = self.get_argument("compartment_id", default=COMPARTMENT_OCID)
        # project_id is optional.
        project_id = self.get_argument("project_id", default=None)
        return self.finish(
            AquaDeploymentApp().list(
                compartment_id=compartment_id, project_id=project_id
            )
        )

    def get_deployment_config(self, model_id: Union[str, List[str]]):
        """
        Retrieves the deployment configuration for one or more Aqua models.

        Parameters
        ----------
        model_id : Union[str, List[str]]
            A single model ID (str) or a list of model IDs (List[str]).

        Returns
        -------
        None
            The function sends the deployment configuration as a response.
        """
        app = AquaDeploymentApp()

        compartment_id = self.get_argument("compartment_id", default=COMPARTMENT_OCID)

        if isinstance(model_id, list):
            # Handle multiple model deployment
            primary_model_id = self.get_argument("primary_model_id", default=None)
            deployment_config = app.get_multimodel_deployment_config(
                model_ids=model_id,
                primary_model_id=primary_model_id,
                compartment_id=compartment_id,
            )
        else:
            # Handle single model deployment
            deployment_config = app.get_deployment_config(model_id=model_id)

        return self.finish(deployment_config)

    def get_recommend_shape(self, model_id: str):
        """
        Retrieves the valid shape and deployment parameter configuration for one Aqua Model.

        Parameters
        ----------
        model_id : str
            A single model ID (str).

        Returns
        -------
        None
            The function sends the ShapeRecommendReport (generate_table = False) or Rich Diff Table (generate_table = True)
        """
        app = AquaDeploymentApp()

        compartment_id = self.get_argument("compartment_id", default=COMPARTMENT_OCID)

        recommend_report = app.recommend_shape(
            model_id=model_id,
            compartment_id=compartment_id,
            generate_table=False,
        )

        return self.finish(recommend_report)

    def list_shapes(self):
        """
        Lists the valid model deployment shapes.

        Returns
        -------
        List[ComputeShapeSummary]:
            The list of the model deployment shapes.
        """
        compartment_id = self.get_argument("compartment_id", default=COMPARTMENT_OCID)

        return self.finish(
            AquaDeploymentApp().list_shapes(compartment_id=compartment_id)
        )


class AquaDeploymentStreamingInferenceHandler(AquaAPIhandler):
    def _extract_text_from_choice(self, choice: dict) -> str:
        """
        Extract text content from a single choice structure.

        Handles both dictionary-based API responses and object-based SDK responses.
        For dict choices, it checks delta-based streaming fields, message-based
        non-streaming fields, and finally top-level text/content keys.
        For object choices, it inspects `.delta`, `.message`, and top-level
        `.text` or `.content` attributes.

        Parameters
        ----------
        choice : dict
            A choice entry from a model response. It may be:
                - A dict originating from a JSON API response (streaming or non-streaming).
                - An SDK-style object with attributes such as `delta`, `message`,
                `text`, or `content`.

                For dicts, the method checks:
                    • delta → content/text
                    • message → content/text
                    • top-level → text/content

                For objects, the method checks the same fields via attributes.

        Returns
        -------
        str | None:
            The extracted text if present; otherwise None.
        """
        # choice may be a dict or an object
        if isinstance(choice, dict):
            # streaming chunk: {"delta": {"content": "..."}}
            delta = choice.get("delta")
            if isinstance(delta, dict):
                return delta.get("content") or delta.get("text") or None
            # non-streaming: {"message": {"content": "..."}}
            msg = choice.get("message")
            if isinstance(msg, dict):
                return msg.get("content") or msg.get("text")
            # fallback top-level fields
            return choice.get("text") or choice.get("content")
        # object-like choice
        delta = getattr(choice, "delta", None)
        if delta is not None:
            return getattr(delta, "content", None) or getattr(delta, "text", None)
        msg = getattr(choice, "message", None)
        if msg is not None:
            if isinstance(msg, str):
                return msg
            return getattr(msg, "content", None) or getattr(msg, "text", None)
        return getattr(choice, "text", None) or getattr(choice, "content", None)

    def _extract_text_from_chunk(self, chunk: dict) -> str:
        """
        Extract text content from a model response chunk.

        Supports both dict-form chunks (streaming or non-streaming) and SDK-style
        object chunks. When choices are present, extraction is delegated to
        `_extract_text_from_choice`. If no choices exist, top-level text/content
        fields or attributes are used.

        Parameters
        ----------
        chunk : dict
            A chunk returned from a model stream or full response. It may be:
            - A dict containing a `choices` list or top-level text/content fields.
            - An SDK-style object with a `choices` attribute or top-level
              `text`/`content` attributes.

            If `choices` is present, the method extracts text from the first
            choice using `_extract_text_from_choice`. Otherwise, it falls back
            to top-level text/content.
        Returns
        -------
        str
            The extracted text if present; otherwise None.
        """
        if chunk:
            if isinstance(chunk, dict):
                choices = chunk.get("choices") or []
                if choices:
                    return self._extract_text_from_choice(choices[0])
                # fallback top-level
                return chunk.get("text") or chunk.get("content")
            # object-like chunk
            choices = getattr(chunk, "choices", None)
            if choices:
                return self._extract_text_from_choice(choices[0])
            return getattr(chunk, "text", None) or getattr(chunk, "content", None)

    def _get_model_deployment_response(self, model_deployment_id: str, payload: dict):
        """
        Returns the model deployment inference response in a streaming fashion.

        This method connects to the specified model deployment endpoint and
        streams the inference output back to the caller, handling both text
        and chat completion endpoints depending on the route override.

        Parameters
        ----------
        model_deployment_id : str
            The OCID of the model deployment to invoke.
            Example: 'ocid1.datasciencemodeldeployment.iad.oc1.xxxyz'

        payload : dict
            Dictionary containing the model inference parameters.
            Same example for text completions:
                {
                    "max_tokens": 1024,
                    "temperature": 0.5,
                    "prompt": "what are some good skills deep learning expert. Give us some tips on how to structure interview with some coding example?",
                    "top_p": 0.4,
                    "top_k": 100,
                    "model": "odsc-llm",
                    "frequency_penalty": 1,
                    "presence_penalty": 1,
                    "stream": true
                }

        route_override_header : Optional[str]
            Optional override for the inference route, used for routing between
            different endpoint types (e.g., chat vs. text completions).
            Example: '/v1/chat/completions'

        Returns
        -------
        Generator[str]
            A generator that yields strings of the model's output as they are received.

        Raises
        ------
        HTTPError
            If the request to the model deployment fails or if streaming cannot be established.
        """

        model_deployment = AquaDeploymentApp().get(model_deployment_id)
        endpoint = model_deployment.endpoint + "/predictWithResponseStream/v1"

        required_keys = ["endpoint_type", "prompt", "model"]
        missing = [k for k in required_keys if k not in payload]

        if missing:
            raise HTTPError(400, f"Missing required payload keys: {', '.join(missing)}")

        endpoint_type = payload["endpoint_type"]
        aqua_client = OpenAI(base_url=endpoint)

        allowed = {
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "n",
            "presence_penalty",
            "frequency_penalty",
            "logprobs",
            "user",
            "echo",
        }
        responses_allowed = {"temperature", "top_p"}

        # normalize and filter
        if payload.get("stop") == []:
            payload["stop"] = None

        encoded_image = "NA"
        if "encoded_image" in payload:
            encoded_image = payload["encoded_image"]

        model = payload.pop("model")
        filtered = {k: v for k, v in payload.items() if k in allowed}
        responses_filtered = {
            k: v for k, v in payload.items() if k in responses_allowed
        }

        if (
            endpoint_type == PredictEndpoints.CHAT_COMPLETIONS_ENDPOINT
            and encoded_image == "NA"
        ):
            try:
                api_kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": payload["prompt"]}],
                    "stream": True,
                    **filtered,
                }
                if "chat_template" in payload:
                    chat_template = payload.pop("chat_template")
                    api_kwargs["extra_body"] = {"chat_template": chat_template}

                stream = aqua_client.chat.completions.create(**api_kwargs)

                for chunk in stream:
                    if chunk:
                        piece = self._extract_text_from_chunk(chunk)
                        if piece:
                            yield piece
            except ExtendedRequestError as ex:
                raise HTTPError(400, str(ex)) from ex
            except Exception as ex:
                raise HTTPError(500, str(ex)) from ex

        elif (
            endpoint_type == PredictEndpoints.CHAT_COMPLETIONS_ENDPOINT
            and encoded_image != "NA"
        ):
            file_type = payload.pop("file_type")
            if file_type.startswith("image"):
                api_kwargs = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": payload["prompt"]},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"{encoded_image}"},
                                },
                            ],
                        }
                    ],
                    "stream": True,
                    **filtered,
                }

                # Add chat_template for image-based chat completions
                if "chat_template" in payload:
                    chat_template = payload.pop("chat_template")
                    api_kwargs["extra_body"] = {"chat_template": chat_template}

                response = aqua_client.chat.completions.create(**api_kwargs)

            elif file_type.startswith("audio"):
                api_kwargs = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": payload["prompt"]},
                                {
                                    "type": "audio_url",
                                    "audio_url": {"url": f"{encoded_image}"},
                                },
                            ],
                        }
                    ],
                    "stream": True,
                    **filtered,
                }

                # Add chat_template for audio-based chat completions
                if "chat_template" in payload:
                    chat_template = payload.pop("chat_template")
                    api_kwargs["extra_body"] = {"chat_template": chat_template}

                response = aqua_client.chat.completions.create(**api_kwargs)
            try:
                for chunk in response:
                    piece = self._extract_text_from_chunk(chunk)
                    if piece:
                        yield piece
            except ExtendedRequestError as ex:
                raise HTTPError(400, str(ex)) from ex
            except Exception as ex:
                raise HTTPError(500, str(ex)) from ex
        elif endpoint_type == PredictEndpoints.TEXT_COMPLETIONS_ENDPOINT:
            try:
                for chunk in aqua_client.completions.create(
                    prompt=payload["prompt"], stream=True, model=model, **filtered
                ):
                    if chunk:
                        piece = self._extract_text_from_chunk(chunk)
                        if piece:
                            yield piece
            except ExtendedRequestError as ex:
                raise HTTPError(400, str(ex)) from ex
            except Exception as ex:
                raise HTTPError(500, str(ex)) from ex

        elif endpoint_type == PredictEndpoints.RESPONSES:
            kwargs = {"model": model, "input": payload["prompt"], "stream": True}

            if "temperature" in responses_filtered:
                kwargs["temperature"] = responses_filtered["temperature"]
            if "top_p" in responses_filtered:
                kwargs["top_p"] = responses_filtered["top_p"]

            response = aqua_client.responses.create(**kwargs)
            try:
                for chunk in response:
                    if chunk:
                        piece = self._extract_text_from_chunk(chunk)
                        if piece:
                            yield piece
            except ExtendedRequestError as ex:
                raise HTTPError(400, str(ex)) from ex
            except Exception as ex:
                raise HTTPError(500, str(ex)) from ex
        else:
            raise HTTPError(400, f"Unsupported endpoint_type: {endpoint_type}")

    @handle_exceptions
    def post(self, model_deployment_id):
        """
        Handles streaming inference request for the Active Model Deployments
        Raises
        ------
        HTTPError
            Raises HTTPError if inputs are missing or are invalid
        """
        try:
            input_data = self.get_json_body()
        except Exception as ex:
            raise HTTPError(400, Errors.INVALID_INPUT_DATA_FORMAT) from ex

        if not input_data:
            raise HTTPError(400, Errors.NO_INPUT_DATA)

        prompt = input_data.get("prompt")
        messages = input_data.get("messages")

        if not prompt and not messages:
            raise HTTPError(
                400, Errors.MISSING_REQUIRED_PARAMETER.format("prompt/messages")
            )
        if not input_data.get("model"):
            raise HTTPError(400, Errors.MISSING_REQUIRED_PARAMETER.format("model"))
        self.set_header("Content-Type", "text/event-stream")
        response_gen = self._get_model_deployment_response(
            model_deployment_id, input_data
        )
        try:
            for chunk in response_gen:
                self.write(chunk)
                self.flush()
            self.finish()
        except Exception as ex:
            self.set_status(getattr(ex, "status_code", 500))
            self.write({"message": "Error occurred", "reason": str(ex)})
            self.finish()


class AquaDeploymentParamsHandler(AquaAPIhandler):
    """Handler for Aqua deployment params REST APIs.

    Methods
    -------
    get(self, model_id)
        Retrieves a list of model deployment parameters.
    post(self, *args, **kwargs)
        Validates parameters for the given model id.
    """

    @handle_exceptions
    def get(self, model_id):
        """Handle GET request."""
        instance_shape = self.get_argument("instance_shape")
        gpu_count = self.get_argument("gpu_count", default=None)
        return self.finish(
            AquaDeploymentApp().get_deployment_default_params(
                model_id=model_id, instance_shape=instance_shape, gpu_count=gpu_count
            )
        )

    @handle_exceptions
    def post(self, *args, **kwargs):  # noqa: ARG002
        """Handles post request for the deployment param handler API.

        Raises
        ------
        HTTPError
            Raises HTTPError if inputs are missing or are invalid.
        """
        try:
            input_data = self.get_json_body()
        except Exception as ex:
            raise HTTPError(400, Errors.INVALID_INPUT_DATA_FORMAT) from ex

        if not input_data:
            raise HTTPError(400, Errors.NO_INPUT_DATA)

        model_id = input_data.get("model_id")
        if not model_id:
            raise HTTPError(400, Errors.MISSING_REQUIRED_PARAMETER.format("model_id"))

        params = input_data.get("params")
        container_family = input_data.get("container_family")
        return self.finish(
            AquaDeploymentApp().validate_deployment_params(
                model_id=model_id,
                params=params,
                container_family=container_family,
            )
        )


class AquaModelListHandler(AquaAPIhandler):
    """Handler for Aqua model list params REST APIs.

    Methods
    -------
    get(self, *args, **kwargs)
        Validates parameters for the given model id.
    """

    @handle_exceptions
    def get(self, model_deployment_id):
        """
        Handles get model list for the Active Model Deployment
        Raises
        ------
        HTTPError
            Raises HTTPError if inputs are missing or are invalid
        """

        self.set_header("Content-Type", "application/json")
        endpoint: str = ""
        model_deployment = AquaDeploymentApp().get(model_deployment_id)
        endpoint = model_deployment.endpoint.rstrip("/") + "/predict/v1/models"
        aqua_client = Client(endpoint=endpoint)
        try:
            list_model_result = aqua_client.fetch_data()
            return self.finish(list_model_result)
        except Exception as ex:
            error_type = type(ex).__name__
            error_message = (
                f"Error fetching data from endpoint '{endpoint}' [{error_type}]: {ex}"
            )
            logger.error(
                error_message, exc_info=True
            )  # Log with stack trace for diagnostics
            raise HTTPError(500, error_message) from ex


__handlers__ = [
    ("deployments/?([^/]*)/params", AquaDeploymentParamsHandler),
    ("deployments/config/?([^/]*)", AquaDeploymentHandler),
    ("deployments/shapes/?([^/]*)", AquaDeploymentHandler),
    ("deployments/recommend_shapes/?([^/]*)", AquaDeploymentHandler),
    ("deployments/?([^/]*)", AquaDeploymentHandler),
    ("deployments/?([^/]*)/activate", AquaDeploymentHandler),
    ("deployments/?([^/]*)/deactivate", AquaDeploymentHandler),
    ("inference/stream/?([^/]*)", AquaDeploymentStreamingInferenceHandler),
    ("deployments/models/list/?([^/]*)", AquaModelListHandler),
]
