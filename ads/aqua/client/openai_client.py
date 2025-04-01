#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import logging
from typing import Any, Dict, Optional

import httpx
from git import Union

from ads.aqua.client.client import get_async_httpx_client, get_httpx_client

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = httpx.Timeout(timeout=600, connect=5.0)
DEFAULT_MAX_RETRIES = 2

try:
    import openai
except ImportError as e:
    raise ModuleNotFoundError(
        "The custom OpenAI client requires the `openai-python` package. "
        "Please install it with `pip install openai`."
    ) from e


class AquaOpenAIMixin:
    """
    Mixin that provides common logic to patch request headers and URLs
    for both synchronous and asynchronous clients.
    """

    def _patch_route(self, original_path: str) -> str:
        """
        Determine the appropriate route header based on the original URL path.

        Args:
            original_path (str): The original URL path.

        Returns:
            str: The route header value.
        """
        route = (
            original_path.lower()
            .rstrip("/")
            .replace(self.base_url.path.lower().rstrip("/"), "")
        )
        return f"/v1{route}" if route else ""

    def _patch_streaming(self, request: httpx.Request) -> None:
        """
        Set the 'enable-streaming' header based on whether the JSON request body contains
        a 'stream': true parameter.

        If the Content-Type is JSON, the request body is parsed. If the key 'stream' is set to True,
        the header 'enable-streaming' is set to "true". Otherwise, it is set to "false".
        If parsing fails, a warning is logged and the default value remains "false".

        Args:
            request (httpx.Request): The outgoing HTTP request.
        """
        streaming_enabled = "false"
        content_type = request.headers.get("Content-Type", "")
        if "application/json" in content_type and request.content:
            try:
                body_str = (
                    request.content.decode("utf-8")
                    if isinstance(request.content, bytes)
                    else request.content
                )
                data = json.loads(body_str)
                if data.get("stream") is True:
                    streaming_enabled = "true"
            except Exception as e:
                logger.exception("Failed to parse JSON from request body: %s", e)
        request.headers.setdefault("enable-streaming", streaming_enabled)
        logger.debug(
            "Patched streaming header to: %s", request.headers["enable-streaming"]
        )

    def _patch_headers(self, request: httpx.Request) -> None:
        """
        Patch the headers of the request by setting the 'enable-streaming' and 'route' headers.

        Args:
            request (httpx.Request): The HTTP request to patch.
        """
        self._patch_streaming(request)
        request.headers.setdefault("route", self._patch_route(request.url.path))
        logger.debug("Patched route header to: %s", request.headers["route"])

    def _prepare_request_common(self, request: httpx.Request) -> None:
        """
        Prepare the HTTP request by patching headers and normalizing the URL path.

        This method:
          1. Automatically sets the 'enable-streaming' header based on the request body.
          2. Determines the 'route' header based on the original URL path using OCID-based extraction.
          3. Rewrites the URL path to always end with '/predict' based on the deployment base.

        Args:
            request (httpx.Request): The outgoing HTTP request.
        """
        # Patches the headers
        logger.debug("Original headers: %s", request.headers)
        self._patch_headers(request)
        logger.debug("Headers after patching: %s", request.headers)

        # Patches the URL
        request.url = self.base_url.copy_with(path=self.base_url.path.rstrip("/"))


class OpenAI(openai.OpenAI, AquaOpenAIMixin):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        base_url: Optional[Union[str, httpx.URL]] = None,
        websocket_base_url: Optional[Union[str, httpx.URL]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Optional[Dict[str, str]] = None,
        default_query: Optional[Dict[str, object]] = None,
        http_client: Optional[httpx.Client] = None,
        http_client_kwargs: Optional[Dict[str, Any]] = None,
        _strict_response_validation: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Construct a new synchronous OpenAI client instance.

        If no http_client is provided, one will be automatically created using ads.aqua.get_httpx_client().

        Args:
            api_key (str, optional): API key for authentication. Defaults to env variable OPENAI_API_KEY.
            organization (str, optional): Organization ID. Defaults to env variable OPENAI_ORG_ID.
            project (str, optional): Project ID. Defaults to env variable OPENAI_PROJECT_ID.
            base_url (str | httpx.URL, optional): Base URL for the API.
            websocket_base_url (str | httpx.URL, optional): Base URL for WebSocket connections.
            timeout (float | httpx.Timeout, optional): Timeout for API requests.
            max_retries (int, optional): Maximum number of retries for API requests.
            default_headers (dict[str, str], optional): Additional headers.
            default_query (dict[str, object], optional): Additional query parameters.
            http_client (httpx.Client, optional): Custom HTTP client; if not provided, one will be auto-created.
            http_client_kwargs (dict[str, Any], optional): Extra kwargs for auto-creating the HTTP client.
            _strict_response_validation (bool, optional): Enable strict response validation.
            **kwargs: Additional keyword arguments passed to the parent __init__.
        """
        if http_client is None:
            logger.debug(
                "No http_client provided; auto-creating one using ads.aqua.get_httpx_client()"
            )
            http_client = get_httpx_client(**(http_client_kwargs or {}))
        if not api_key:
            logger.debug("API key not provided; using default placeholder for OCI.")
            api_key = "OCI"

        super().__init__(
            api_key=api_key,
            organization=organization,
            project=project,
            base_url=base_url,
            websocket_base_url=websocket_base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
            **kwargs,
        )

    def _prepare_request(self, request: httpx.Request) -> None:
        """
        Prepare the synchronous HTTP request by applying common modifications.

        Args:
            request (httpx.Request): The outgoing HTTP request.
        """
        self._prepare_request_common(request)


class AsyncOpenAI(openai.AsyncOpenAI, AquaOpenAIMixin):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        base_url: Optional[Union[str, httpx.URL]] = None,
        websocket_base_url: Optional[Union[str, httpx.URL]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Optional[Dict[str, str]] = None,
        default_query: Optional[Dict[str, object]] = None,
        http_client: Optional[httpx.Client] = None,
        http_client_kwargs: Optional[Dict[str, Any]] = None,
        _strict_response_validation: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Construct a new asynchronous AsyncOpenAI client instance.

        If no http_client is provided, one will be automatically created using
        ads.aqua.get_async_httpx_client().

        Args:
            api_key (str, optional): API key for authentication. Defaults to env variable OPENAI_API_KEY.
            organization (str, optional): Organization ID.
            project (str, optional): Project ID.
            base_url (str | httpx.URL, optional): Base URL for the API.
            websocket_base_url (str | httpx.URL, optional): Base URL for WebSocket connections.
            timeout (float | httpx.Timeout, optional): Timeout for API requests.
            max_retries (int, optional): Maximum number of retries for API requests.
            default_headers (dict[str, str], optional): Additional headers.
            default_query (dict[str, object], optional): Additional query parameters.
            http_client (httpx.AsyncClient, optional): Custom asynchronous HTTP client; if not provided, one will be auto-created.
            http_client_kwargs (dict[str, Any], optional): Extra kwargs for auto-creating the HTTP client.
            _strict_response_validation (bool, optional): Enable strict response validation.
            **kwargs: Additional keyword arguments passed to the parent __init__.
        """
        if http_client is None:
            logger.debug(
                "No async http_client provided; auto-creating one using ads.aqua.get_async_httpx_client()"
            )
            http_client = get_async_httpx_client(**(http_client_kwargs or {}))
        if not api_key:
            logger.debug("API key not provided; using default placeholder for OCI.")
            api_key = "OCI"

        super().__init__(
            api_key=api_key,
            organization=organization,
            project=project,
            base_url=base_url,
            websocket_base_url=websocket_base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
            **kwargs,
        )

    async def _prepare_request(self, request: httpx.Request) -> None:
        """
        Asynchronously prepare the HTTP request by applying common modifications.

        Args:
            request (httpx.Request): The outgoing HTTP request.
        """
        self._prepare_request_common(request)
