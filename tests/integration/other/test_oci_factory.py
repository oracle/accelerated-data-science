#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

# The modification on oci import in ads.__init__.py will cause the following
# "import oci" to re-import oci service modules.
# The OCIClientFactory has a cached version of the clients,
# which do not equal to the clients from the re-imported oci modules.
import oci
from ads.common import auth as authutil
from ads.common import oci_client as oc


def test_object_storage():
    auth = authutil.default_signer()
    assert str(oc.OCIClientFactory(**auth).object_storage.__class__) == str(
        oci.object_storage.ObjectStorageClient
    )


def test_object_storage_with_kwargs():
    auth = authutil.default_signer(client_kwargs={"timeout": 60000})
    assert str(oc.OCIClientFactory(**auth).object_storage.__class__) == str(
        oci.object_storage.ObjectStorageClient
    )


def test_data_science_client():
    auth = authutil.default_signer()
    assert str(oc.OCIClientFactory(**auth).data_science.__class__) == str(
        oci.data_science.DataScienceClient
    )


def test_identity():
    auth = authutil.default_signer()
    assert str(oc.OCIClientFactory(**auth).identity.__class__) == str(
        oci.identity.IdentityClient
    )


def test_dataflow():
    auth = authutil.default_signer()
    assert str(oc.OCIClientFactory(**auth).dataflow.__class__) == str(
        oci.data_flow.DataFlowClient
    )


def test_secret():
    auth = authutil.default_signer()
    assert str(oc.OCIClientFactory(**auth).secret.__class__) == str(
        oci.secrets.SecretsClient
    )


def test_vault():
    auth = authutil.default_signer()
    assert str(oc.OCIClientFactory(**auth).vault.__class__) == str(
        oci.vault.VaultsClient
    )


def test_ai_language():
    auth = authutil.default_signer()
    assert str(oc.OCIClientFactory(**auth).ai_language.__class__) == str(
        oci.ai_language.AIServiceLanguageClient
    )
