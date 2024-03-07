#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging

from ads import set_auth
from ads.common import oci_client as oc
from ads.common.auth import default_signer
from ads.config import OCI_RESOURCE_PRINCIPAL_VERSION


logger = logging.getLogger(__name__)
class TelemetryBase:
    """Base class for Telemetry Client."""

    def __init__(self, bucket: str, namespace: str = None) -> None:
        """Initializes the telemetry client.

        Parameters
        ----------
        bucket : str
            OCI object storage bucket name storing the telemetry objects.
        namespace : str, optional
            Namespace of the OCI object storage bucket, by default None.
        """
        if OCI_RESOURCE_PRINCIPAL_VERSION:
            set_auth("resource_principal")
        self._auth = default_signer()
        self.os_client = oc.OCIClientFactory(**self._auth).object_storage
        self.bucket = bucket
        self._namespace = namespace
        self._service_endpoint = None
        logger.debug(f"Initialized Telemetry. Namespace: {self.namespace}, Bucket: {self.bucket}")


    @property
    def namespace(self) -> str:
        """Gets the namespace of the object storage from the tenancy.

        Returns
        -------
        str
            The namespace of the tenancy.
        """
        if not self._namespace:
            self._namespace = self.os_client.get_namespace().data
        return self._namespace

    @property
    def service_endpoint(self):
        """Gets the tenancy-specific endpoint.

        Returns
        -------
        str
            Tenancy-specific endpoint.
        """
        if not self._service_endpoint:
            self._service_endpoint = self.os_client.base_client.endpoint
        return self._service_endpoint
