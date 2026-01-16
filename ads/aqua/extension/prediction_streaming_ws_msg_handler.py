#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
from typing import List, Union

from tornado.web import HTTPError

from ads.aqua.client.client import ExtendedRequestError
from ads.aqua.client.openai_client import OpenAI
from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.common.enums import PredictEndpoints
from ads.aqua.extension.aqua_ws_msg_handler import AquaWSMsgHandler
from ads.aqua.extension.errors import Errors
from ads.aqua.extension.models.ws_models import (
    PredictionStreamResponse,
    RequestResponseType,
)
from ads.aqua.modeldeployment import AquaDeploymentApp


class AquaPredictionStreamingWSMsgHandler(AquaWSMsgHandler):
    def __init__(self, message: Union[str, bytes]):
        super().__init__(message)

    def set_stream_callback(self, callback):
        """Sets the callback function to handle streaming chunks."""
        self.stream_callback = callback

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

    # def _extract_text_from_chunk(self, chunk: dict) -> str:
    #     """
    #     Extract text content from a model response chunk.
    #
    #     Supports both dict-form chunks (streaming or non-streaming) and SDK-style
    #     object chunks. When choices are present, extraction is delegated to
    #     `_extract_text_from_choice`. If no choices exist, top-level text/content
    #     fields or attributes are used.
    #
    #     Parameters
    #     ----------
    #     chunk : dict
    #         A chunk returned from a model stream or full response. It may be:
    #         - A dict containing a `choices` list or top-level text/content fields.
    #         - An SDK-style object with a `choices` attribute or top-level
    #           `text`/`content` attributes.
    #
    #         If `choices` is present, the method extracts text from the first
    #         choice using `_extract_text_from_choice`. Otherwise, it falls back
    #         to top-level text/content.
    #     Returns
    #     -------
    #     str
    #         The extracted text if present; otherwise None.
    #     """
    #     if chunk:
    #         if isinstance(chunk, dict):
    #             choices = chunk.get("choices") or []
    #             if choices:
    #                 return self._extract_text_from_choice(choices[0])
    #             # fallback top-level
    #             return chunk.get("text") or chunk.get("content")
    #         # object-like chunk
    #         choices = getattr(chunk, "choices", None)
    #         if choices:
    #             return self._extract_text_from_choice(choices[0])
    #         return getattr(chunk, "text", None) or getattr(chunk, "content", None)

    def _extract_text_from_chunk(self, chunk: dict) -> str:
        if chunk:
            # 1. Handle Dicts (JSON)
            if isinstance(chunk, dict):
                choices = chunk.get("choices") or []
                if choices:
                    return self._extract_text_from_choice(choices[0])
                return chunk.get("text") or chunk.get("content")

            # 2. Handle Objects (SDK Responses)

            # CASE A: Check for top-level 'delta' string (Fixes ResponseTextDeltaEvent)
            # ResponseTextDeltaEvent(delta=' into', ...)
            delta_val = getattr(chunk, "delta", None)
            if delta_val and isinstance(delta_val, str):
                return delta_val

            # CASE B: Standard OpenAI Object (chunk.choices[0].delta.content)
            choices = getattr(chunk, "choices", None)
            if choices:
                return self._extract_text_from_choice(choices[0])

            # CASE C: Fallback
            return getattr(chunk, "text", None) or getattr(chunk, "content", None)
        return None

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

        required_keys = ["prompt","endpoint_type", "model"]
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

    @staticmethod
    def get_message_types() -> List[RequestResponseType]:
        return [RequestResponseType.PredictionStream]

    @handle_exceptions
    def process(self) -> PredictionStreamResponse:
        request = json.loads(self.message)
        if request.get("kind") == "PredictionStream":
            params = request.get("params")
            prompt = params.get("prompt")
            messages = params.get("messages")
            model_deployment_id = request.get("model_deployment_id")
            if not prompt and not messages:
                raise HTTPError(
                    400, Errors.MISSING_REQUIRED_PARAMETER.format("prompt/messages")
                )
            if not params.get("model"):
                raise HTTPError(400, Errors.MISSING_REQUIRED_PARAMETER.format("model"))
            response_gen = self._get_model_deployment_response(
                model_deployment_id, params
            )
            try:
                for chunk in response_gen:
                    response = PredictionStreamResponse(
                        message_id=request.get("message_id"),
                        kind=RequestResponseType.PredictionStream,
                        data=chunk,
                    )
                    if self.stream_callback:
                        self.stream_callback(response)

                return PredictionStreamResponse(
                    message_id=request.get("message_id"),
                    kind=RequestResponseType.PredictionStream,
                    data="[DONE]",  # Or "[DONE]" depending on your frontend logic
                )
            except Exception as ex:
                ex = {
                    "message": "Error occurred",
                    "reason": str(ex),
                    "status_code": 500,
                }
                response = PredictionStreamResponse(
                    data=ex,
                    kind=RequestResponseType.PredictionStream,
                    message_id=request.get("message_id"),
                )
                return response
