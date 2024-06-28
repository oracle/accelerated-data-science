#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
from typing import Any, Dict, Generator, Optional, Tuple, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ads.common import auth as authutil

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 3
TIMEOUT = (600, 900)  # (connect_timeout, read_timeout)


class ModelInvoker:
    """
    A class to invoke models via HTTP requests with retry logic.

    Attributes
    ----------
    endpoint : str
        The URL endpoint to send the request.
    retries : int
        The number of retry attempts for the request.
    backoff_factor : float
        The factor to determine the delay between retries.
    timeout : Union[float, Tuple[float, float]]
        The timeout setting for the HTTP request.
    auth : Any
        The authentication signer for the requests.
    """

    def __init__(
        self,
        endpoint: str,
        retries: Optional[int] = DEFAULT_RETRIES,
        backoff_factor: Optional[float] = DEFAULT_BACKOFF_FACTOR,
        timeout: Optional[Union[float, Tuple[float, float]]] = None,
        auth: Optional[Any] = None,
        **kwargs: Dict,
    ) -> None:
        self.auth = auth or authutil.default_signer()
        self.endpoint = endpoint
        self.retries = retries or DEFAULT_RETRIES
        self.backoff_factor = backoff_factor or DEFAULT_BACKOFF_FACTOR
        self.session = self._create_session_with_retries(
            self.retries, self.backoff_factor
        )
        self.timeout = timeout or TIMEOUT

    def _create_session_with_retries(
        self, retries: int, backoff_factor: float
    ) -> requests.Session:
        """
        Creates a requests Session with a mounted HTTPAdapter for retry logic.

        Parameters
        ----------
        retries : int
            The number of retry attempts for the request.
        backoff_factor : float
            The factor to determine the delay between retries.

        Returns
        -------
        requests.Session
            The configured session for HTTP requests.
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

    def invoke(
        self, params: Dict[str, Any], headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Sends a POST request to the configured endpoint with the specified parameters,
        expecting a JSON response.

        Parameters
        ----------
        params : Dict[str, Any]
            Additional parameters for the model.
        headers : Optional[Dict[str, str]]
            Additional headers for the request.

        Returns
        -------
        Dict[str, Any]
            A dictionary representation of the JSON-decoded response from the model's API.

        Raises
        ------
        requests.RequestException
            An exception indicating an error occurred during the request to the model's API.
        json.JSONDecodeError
            An exception indicating an error in decoding the JSON response.
        """

        headers = {
            **{
                "Content-Type": "application/json",
            },
            **(headers or {}),
        }

        try:
            response = self.session.post(
                self.endpoint,
                auth=self.auth["signer"],
                headers=headers,
                json=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Error decoding JSON from response: {str(e)}", e.doc, e.pos
            )

    def stream(
        self, params: Dict[str, Any], headers: Optional[Dict[str, str]] = None
    ) -> Generator[Any, Any, Any]:
        """
        Sends a POST request to the configured endpoint with the specified parameters,
        expecting a streaming response.

        Parameters
        ----------
        params : Dict[str, Any]
            Additional parameters for the model.
        headers : Optional[Dict[str, str]]
            Additional headers for the request.

        Yields
        ------
        str
            Each line of the response as it is received.

        Raises
        ------
        requests.RequestException
            An exception indicating an error occurred during the request to the model's API.
        json.JSONDecodeError
            An exception indicating an error in decoding the JSON response.
        """
        headers = {
            **{
                "Content-Type": "application/json",
                "enable-streaming": "true",
            },
            **(headers or {}),
        }

        try:
            response = self.session.post(
                self.endpoint,
                auth=self.auth["signer"],
                headers=headers,
                json=params,
                timeout=self.timeout,
                stream=True,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    yield line.decode("utf-8")

        except requests.RequestException as e:
            raise
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Error decoding JSON from response: {str(e)}", e.doc, e.pos
            )
