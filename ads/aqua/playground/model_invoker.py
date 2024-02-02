#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ads.common.auth import default_signer

DEFAULT_RETRIES = 3

# The amount of time to wait between retry attempts for failed request.
DEFAULT_BACKOFF_FACTOR = 0.3


class ModelInvoker:
    """
    A class to invoke models via HTTP requests with retry logic.


    Attributes
    ----------
    endpoint (str): The URL endpoint to send the request.
    prompt (str): The prompt to send in the request body.
    params (dict): Additional parameters for the model.
    retries (int): The number of retry attempts for the request.
    backoff_factor (float): The factor to determine the delay between retries.
    """

    def __init__(
        self,
        endpoint: str,
        prompt: str,
        params: dict,
        retries: int = DEFAULT_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        auth=None,
    ):
        self.auth = auth or default_signer()
        self.endpoint = endpoint
        self.prompt = prompt
        self.params = params
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.session = self._create_session_with_retries(retries, backoff_factor)

    def _create_session_with_retries(
        self, retries: int, backoff_factor: float
    ) -> requests.Session:
        """
        Creates a requests Session with a mounted HTTPAdapter for retry logic.

        Returns
        -------
        session (requests.Session): The configured session for HTTP requests.
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=backoff_factor,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        return session

    def invoke(self):
        """
        The generator that invokes the model endpoint with retries and streams the response.

        Yields
        ------
        line (str): A line of the streamed response.
        """
        headers = {
            "Content-Type": "application/json",
            "enable-streaming": "true",
        }

        # print({"prompt": self.prompt, **self.params})

        try:
            response = self.session.post(
                self.endpoint,
                auth=self.auth["signer"],
                headers=headers,
                json={"prompt": self.prompt, **self.params},
                stream=True,
            )

            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    yield line.decode("utf-8")

        except requests.RequestException as e:
            yield json.dumps({"object": "error", "message": str(e)})
