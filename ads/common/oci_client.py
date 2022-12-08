#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging

from oci.ai_language import AIServiceLanguageClient
from oci.data_flow import DataFlowClient
from oci.data_labeling_service import DataLabelingManagementClient
from oci.data_labeling_service_dataplane import DataLabelingClient
from oci.data_science import DataScienceClient
from oci.identity import IdentityClient
from oci.object_storage import ObjectStorageClient
from oci.secrets import SecretsClient
from oci.vault import VaultsClient

logger = logging.getLogger(__name__)


class OCIClientFactory:

    """
    A factory class to create OCI client objects. The constructor takes in config, signer and client_kwargs. `client_kwargs` is passed
    to the client constructor as key word argutments.

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

    def __init__(self, config={}, signer=None, client_kwargs=None):
        self.config = config
        self.signer = signer
        self.client_kwargs = client_kwargs

    def _client_impl(self, client):
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
        }

        assert (
            client in client_map
        ), f"Invalid client name. Client name not found in {client_map.keys()}"
        return client_map[client]

    def _validate_auth_param(self, auth):
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
