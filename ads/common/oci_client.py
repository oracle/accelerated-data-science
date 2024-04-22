#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging

import oci.artifacts
from oci.ai_language import AIServiceLanguageClient
from oci.artifacts import ArtifactsClient
from oci.data_catalog import DataCatalogClient
from oci.data_flow import DataFlowClient
from oci.data_labeling_service import DataLabelingManagementClient
from oci.data_labeling_service_dataplane import DataLabelingClient
from oci.data_science import DataScienceClient
from oci.identity import IdentityClient
from oci.marketplace import MarketplaceClient
from oci.object_storage import ObjectStorageClient
from oci.resource_search import ResourceSearchClient
from oci.secrets import SecretsClient
from oci.vault import VaultsClient
from oci.logging import LoggingManagementClient
from oci.core import VirtualNetworkClient
from oci.limits import LimitsClient

logger = logging.getLogger(__name__)


class OCIClientFactory:
    """
    A factory class to create OCI client objects. The constructor takes in config, signer and client_kwargs. `client_kwargs` is passed
    to the client constructor as key word arguments.

    Examples
    --------
    from ads.common import auth as authutil
    from ads.common import oci_client as oc

    auth = authutil.default_signer()
    oc.OCIClientFactory(**auth).object_storage # Creates Object storage client

    auth = authutil.default_signer({"timeout": 6000})
    oc.OCIClientFactory(**auth).object_storage # Creates Object storage client with timeout set to 6000

    auth = authutil.api_keys(config="/home/datascience/.oci/config", profile="TEST", {"timeout": 6000})
    oc.OCIClientFactory(**auth).object_storage # Creates Object storage client with timeout set to 6000 using API Key authentication

    auth = authutil.resource_principal({"timeout": 6000})
    oc.OCIClientFactory(**auth).object_storage # Creates Object storage client with timeout set to 6000 using resource principal authentication

    auth = authutil.create_signer("instance_principal")
    oc.OCIClientFactory(**auth).object_storage # Creates Object storage client using instance principal authentication
    """

    def __init__(self, config=None, signer=None, client_kwargs=None):
        if not config:
            config = {}
        self.config = config
        self.signer = signer
        self.client_kwargs = client_kwargs

    @staticmethod
    def _client_impl(client):
        client_map = {
            "object_storage": ObjectStorageClient,
            "data_science": DataScienceClient,
            "dataflow": DataFlowClient,
            "secret": SecretsClient,
            "vault": VaultsClient,
            "identity": IdentityClient,
            "ai_language": AIServiceLanguageClient,
            "data_labeling_dp": DataLabelingClient,
            "data_labeling_cp": DataLabelingManagementClient,
            "resource_search": ResourceSearchClient,
            "data_catalog": DataCatalogClient,
            "logging_management": LoggingManagementClient,
            "virtual_network": VirtualNetworkClient,
            "limits": LimitsClient,
            "marketplace": MarketplaceClient,
            "artifacts": ArtifactsClient,
        }

        assert (
            client in client_map
        ), f"Invalid client name. Client name not found in {client_map.keys()}"
        return client_map[client]

    @staticmethod
    def _validate_auth_param(auth):
        if not isinstance(auth, dict):
            raise ValueError("auth parameter should be of type dictionary")
        if "config" in auth and not isinstance(auth["config"], dict):
            raise ValueError("auth[config] should be of type dict")
        if "signer" in auth and auth["signer"] is None and len(auth["config"]) == 0:
            raise ValueError(
                "Signer is None and auth[config] is empty. Either assign config or if you are using resource principal, set resource principal signer to signer and {} for config"
            )
        return True

    def create_client(self, client_name):
        assert (
            client_name is not None and client_name != ""
        ), "Client name cannot be empty"
        client = (
            self._client_impl(client_name)
            if isinstance(client_name, str)
            else client_name
        )
        kwargs = self.client_kwargs or dict()
        return client(config=self.config, signer=self.signer, **kwargs)

    @property
    def object_storage(self):
        return self.create_client("object_storage")

    @property
    def identity(self):
        return self.create_client("identity")

    @property
    def data_science(self):
        return self.create_client("data_science")

    @property
    def dataflow(self):
        return self.create_client("dataflow")

    @property
    def secret(self):
        return self.create_client("secret")

    @property
    def vault(self):
        return self.create_client("vault")

    @property
    def ai_language(self):
        return self.create_client("ai_language")

    @property
    def data_labeling_cp(self):
        return self.create_client("data_labeling_cp")

    @property
    def data_labeling_dp(self):
        return self.create_client("data_labeling_dp")

    @property
    def resource_search(self):
        return self.create_client("resource_search")

    @property
    def data_catalog(self):
        return self.create_client("data_catalog")

    @property
    def logging_management(self):
        return self.create_client("logging_management")

    @property
    def virtual_network(self):
        return self.create_client("virtual_network")

    @property
    def limits(self):
        return self.create_client("limits")

    @property
    def marketplace(self):
        return self.create_client("marketplace")

    @property
    def artifacts(self) -> oci.artifacts.ArtifactsClient:
        return self.create_client("artifacts")
