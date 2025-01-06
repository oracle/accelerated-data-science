#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import logging
import threading
import traceback
import urllib.parse
from typing import Optional

import oci

from ads.config import DEBUG_TELEMETRY

from .base import TelemetryBase

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
    >>> AQUA_BUCKET_NS = os.environ.get("AQUA_BUCKET_NS", "namespace")
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
    ) -> Optional[int]:
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
        int
            The status code for the telemetry request.
            200: The the object exists for the telemetry request
            404: The the object does not exist for the telemetry request.
            Note that for telemetry purpose, the object does not need to be exist.
            `None` will be returned if the telemetry request failed.
        """
        try:
            if not category or not action:
                raise ValueError("Please specify the category and the action.")
            if detail:
                category, action = f"{category}/{action}", detail
            # Here `endpoint`` is for debugging purpose
            # For some federated/domain users, the `endpoint` may not be a valid URL
            endpoint = f"{self.service_endpoint}/n/{self.namespace}/b/{self.bucket}/o/telemetry/{category}/{action}"
            logger.debug(f"Sending telemetry to endpoint: {endpoint}")

            self.os_client.base_client.user_agent = self._encode_user_agent(**kwargs)
            try:
                response: oci.response.Response = self.os_client.head_object(
                    namespace_name=self.namespace,
                    bucket_name=self.bucket,
                    object_name=f"telemetry/{category}/{action}",
                )
                logger.debug(f"Telemetry status: {response.status}")
                return response.status
            except oci.exceptions.ServiceError as ex:
                if ex.status == 404:
                    return ex.status
                raise ex
        except Exception as e:
            if DEBUG_TELEMETRY:
                logger.error(f"There is an error recording telemetry: {e}")
                traceback.print_exc()
            return None

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
