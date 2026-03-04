#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import logging
import re
from typing import Any, Dict, Optional

import httpx
from git import Union

from ads.aqua.client.client import get_async_httpx_client, get_httpx_client
from ads.common.extended_enum import ExtendedEnum

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


class ModelDeploymentBaseEndpoint(ExtendedEnum):
    """Supported base endpoints for model deployments."""

    PREDICT = "predict"
    PREDICT_WITH_RESPONSE_STREAM = "predictWithResponseStream"


class AquaOpenAIMixin:
    """
    Mixin that provides common logic to patch HTTP request headers and URLs
    for compatibility with the OCI Model Deployment service using the OpenAI API schema.
    """

    def _patch_route(self, original_path: str) -> str:
        """
        Extracts and formats the OpenAI-style route path from a full request path.

        Args:
            original_path (str): The full URL path from the incoming request.

        Returns:
            str: The normalized OpenAI-compatible route path (e.g., '/v1/chat/completions').
        """
        normalized_path = original_path.rstrip("/")

        match = re.search(r"/predict(WithResponseStream)?", normalized_path)
        if not match:
            logger.debug("Route header cannot be resolved from path: %s", original_path)
            return ""

        route_suffix = normalized_path[match.end() :].lstrip("/")
        if not route_suffix:
            logger.warning(
                "Missing OpenAI route suffix after '/predict'. "
                "Expected something like '/v1/completions'."
            )
            return ""

        if not route_suffix.startswith("v"):
            logger.warning(
                "Route suffix does not start with a version prefix (e.g., '/v1'). "
                "This may lead to compatibility issues with OpenAI-style endpoints. "
                "Consider updating the URL to include a version prefix, "
                "such as '/predict/v1' or '/predictWithResponseStream/v1'."
            )
            # route_suffix = f"v1/{route_suffix}"

        return f"/{route_suffix}"

    def _patch_streaming(self, request: httpx.Request) -> None:
        """
        Sets the 'enable-streaming' header based on the JSON request body contents.

        If the request body contains `"stream": true`, the `enable-streaming` header is set to "true".
        Otherwise, it defaults to "false".

        Args:
            request (httpx.Request): The outgoing HTTPX request.
        """
        streaming_enabled = "false"
        content_type = request.headers.get("Content-Type", "")

        if "application/json" in content_type and request.content:
            try:
                body = (
                    request.content.decode("utf-8")
                    if isinstance(request.content, bytes)
                    else request.content
                )
                payload = json.loads(body)
                if payload.get("stream") is True:
                    streaming_enabled = "true"
            except Exception as e:
                logger.exception(
                    "Failed to parse request JSON body for streaming flag: %s", e
                )

        request.headers.setdefault("enable-streaming", streaming_enabled)
        logger.debug("Patched 'enable-streaming' header: %s", streaming_enabled)

    def _patch_headers(self, request: httpx.Request) -> None:
        """
        Patches request headers by injecting OpenAI-compatible values:
        - `enable-streaming` for streaming-aware endpoints
        - `route` for backend routing

        Args:
            request (httpx.Request): The outgoing HTTPX request.
        """
        self._patch_streaming(request)
        route_header = self._patch_route(request.url.path)
        request.headers.setdefault("route", route_header)
        logger.debug("Patched 'route' header: %s", route_header)

    def _patch_url(self) -> httpx.URL:
        """
        Strips any suffixes from the base URL to retain only the `/predict` or `/predictWithResponseStream` path.

        Returns:
            httpx.URL: The normalized base URL with the correct model deployment path.
        """
        base_path = f"{self.base_url.path.rstrip('/')}/"
        match = re.search(r"/predict(WithResponseStream)?/", base_path)
        if match:
            trimmed = base_path[: match.end() - 1]
            return self.base_url.copy_with(path=trimmed)

        logger.debug("Could not determine a valid endpoint from path: %s", base_path)
        return self.base_url

    def _prepare_request_common(self, request: httpx.Request) -> None:
        """
        Common preparation routine for all requests.

        This includes:
        - Patching headers with streaming and routing info.
        - Normalizing the URL path to include only `/predict` or `/predictWithResponseStream`.

        Args:
            request (httpx.Request): The outgoing HTTPX request.
        """
        # Patch headers
        logger.debug("Original headers: %s", request.headers)
        self._patch_headers(request)
        logger.debug("Headers after patching: %s", request.headers)

        # Patch URL
        logger.debug("URL before patching: %s", request.url)
        request.url = self._patch_url()
        logger.debug("URL after patching: %s", request.url)


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
        patch_headers: bool = False,
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
            patch_headers (bool, optional): If True, redirects the requests by modifying the headers.
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

        self.patch_headers = patch_headers

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
        if self.patch_headers:
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
        patch_headers: bool = False,
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
            patch_headers (bool, optional): If True, redirects the requests by modifying the headers.
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

        self.patch_headers = patch_headers

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
        if self.patch_headers:
            self._prepare_request_common(request)
