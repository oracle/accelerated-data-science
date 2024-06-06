#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
from dataclasses import asdict, dataclass, field

import requests

from ads.aqua.app import AquaApp, logger
from ads.aqua.modeldeployment.entities import ModelParams
from ads.common.auth import default_signer
from ads.telemetry import telemetry


@dataclass
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

    prompt: str = None
    model_params: field(default_factory=ModelParams) = None

    @telemetry(entry_point="plugin=inference&action=get_response", name="aqua")
    def get_model_deployment_response(self, endpoint):
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

        params_dict = asdict(self.model_params)
        params_dict = {
            key: value for key, value in params_dict.items() if value is not None
        }
        body = {"prompt": self.prompt, **params_dict}
        request_kwargs = {"json": body, "headers": {"Content-Type": "application/json"}}
        response = requests.post(
            endpoint, auth=default_signer()["signer"], **request_kwargs
        )
        return json.loads(response.content)
