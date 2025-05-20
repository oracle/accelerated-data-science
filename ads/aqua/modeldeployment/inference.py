#!/usr/bin/env python

# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json

from ads.aqua.app import AquaApp
from ads.aqua.modeldeployment.entities import ModelParams
from ads.telemetry import telemetry


class MDInferenceResponse(AquaApp):
    """Contains APIs for Aqua Model deployments Inference.

    Attributes
    ----------

    model_params: Dict
    prompt: string

    Methods
    -------
    get_model_deployment_response(self, **kwargs) -> "String"
        Creates an instance of model deployment via Aqua
    """

    def __init__(self, prompt=None, model_params=None):
        super().__init__()
        self.prompt = prompt
        self.model_params = model_params or ModelParams()

    @staticmethod
    def stream_sanitizer(response):
        for chunk in response.data.raw.stream(1024 * 1024, decode_content=True):
            if not chunk:
                continue

            try:
                decoded = chunk.decode("utf-8").strip()
                if not decoded.startswith("data:"):
                    continue

                data_json = decoded[len("data:") :].strip()
                parsed = json.loads(data_json)
                text = parsed["choices"][0]["text"]
                yield text

            except Exception:
                continue

    @telemetry(entry_point="plugin=inference&action=get_response", name="aqua")
    def get_model_deployment_response(self, model_deployment_id):
        """
        Returns MD inference response

        Parameters
        ----------
        endpoint: str
            MD predict url
        prompt: str
            User prompt.

        model_params: (Dict, optional)
            Model parameters to be associated with the message.
            Currently supported VLLM+OpenAI parameters.

            --model-params '{
                "max_tokens":500,
                "temperature": 0.5,
                "top_k": 10,
                "top_p": 0.5,
                "model": "/opt/ds/model/deployed_model",
                ...}'

        Returns
        -------
        model_response_content
        """

        params_dict = self.model_params.to_dict()
        params_dict = {
            key: value for key, value in params_dict.items() if value is not None
        }
        body = {"prompt": self.prompt, **params_dict}
        response = self.model_deployment_client.predict_with_response_stream(
            model_deployment_id=model_deployment_id, request_body=body
        )

        for chunk in MDInferenceResponse.stream_sanitizer(response):
            yield chunk
