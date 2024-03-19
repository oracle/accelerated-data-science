#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import logging
import threading
import urllib.parse
import requests
from requests import Response
from .base import TelemetryBase
from ads.config import DEBUG_TELEMETRY


logger = logging.getLogger(__name__)


class TelemetryClient(TelemetryBase):
    """Represents a telemetry python client providing functions to record an event.

    Methods
    -------
    record_event(category: str = None, action: str = None, path: str = None, **kwargs) -> None
        Send a head request to generate an event record.
    record_event_async(category: str = None, action: str = None, path: str = None,  **kwargs)
        Starts thread to send a head request to generate an event record.

    Examples
    --------
    >>> import os
    >>> import traceback
    >>> from ads.telemetry.client import TelemetryClient
    >>> AQUA_BUCKET = os.environ.get("AQUA_BUCKET", "service-managed-models")
    >>> AQUA_BUCKET_NS = os.environ.get("AQUA_BUCKET_NS", "ociodscdev")
    >>> telemetry = TelemetryClient(bucket=AQUA_BUCKET, namespace=AQUA_BUCKET_NS)
    >>> telemetry.record_event_async(category="aqua/service/model", action="create") # records create action
    >>> telemetry.record_event_async(category="aqua/service/model/create", action="shape", detail="VM.GPU.A10.1")
    """

    @staticmethod
    def _encode_user_agent(**kwargs):
        message = urllib.parse.urlencode(kwargs)
        return message

    def record_event(
        self, category: str = None, action: str = None, detail: str = None, **kwargs
    ) -> Response:
        """Send a head request to generate an event record.

        Parameters
        ----------
        category: (str)
            Category of the event, which is also the path to the directory containing the object representing the event.
        action: (str)
            Filename of the object representing the event.
        detail: (str)
            Can be used to pass additional values, if required. When set, detail is converted to an action,
            category and action are grouped together for telemetry parsing in the backend.
        **kwargs:
            Can be used to pass additional attributes like value that will be passed in the headers of the request.

        Returns
        -------
        Response
        """
        try:
            if not category or not action:
                raise ValueError("Please specify the category and the action.")
            if detail:
                category, action = f"{category}/{action}", detail
            endpoint = f"{self.service_endpoint}/n/{self.namespace}/b/{self.bucket}/o/telemetry/{category}/{action}"
            headers = {"User-Agent": self._encode_user_agent(**kwargs)}
            logger.debug(f"Sending telemetry to endpoint: {endpoint}")
            signer = self._auth["signer"]
            response = requests.head(endpoint, auth=signer, headers=headers)
            logger.debug(f"Telemetry status code: {response.status_code}")
            return response
        except Exception as e:
            if DEBUG_TELEMETRY:
                logger.error(f"There is an error recording telemetry: {e}")

    def record_event_async(
        self, category: str = None, action: str = None, detail: str = None, **kwargs
    ):
        """Send a head request to generate an event record.

        Parameters
        ----------
        category (str)
            Category of the event, which is also the path to the directory containing the object representing the event.
        action (str)
            Filename of the object representing the event.

        Returns
        -------
        Thread
            A started thread to send a head request to generate an event record.
        """
        thread = threading.Thread(
            target=self.record_event, args=(category, action, detail), kwargs=kwargs
        )
        thread.daemon = True
        thread.start()
        return thread
