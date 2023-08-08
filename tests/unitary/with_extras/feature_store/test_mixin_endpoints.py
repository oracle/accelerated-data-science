#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import ads
from ads.feature_store.mixin.oci_feature_store import OCIFeatureStoreMixin
from ads.common.auth import AuthType

TEST_URL = "https://test.com"


def test_manual_endpoint():
    ads.set_auth(auth=AuthType.API_KEY, client_kwargs={"fs_service_endpoint": TEST_URL})
    client = OCIFeatureStoreMixin.init_client(fs_service_endpoint=TEST_URL)
    assert client.base_client.endpoint == f"{TEST_URL}/20230101"


def test_manual_with_service_endpoint():
    ads.set_auth(
        auth=AuthType.API_KEY,
        client_kwargs={
            "fs_service_endpoint": TEST_URL,
            "service_endpoint": "service.com",
        },
    )
    client = OCIFeatureStoreMixin.init_client(fs_service_endpoint=TEST_URL)
    assert client.base_client.endpoint == f"{TEST_URL}/20230101"


def test_service_endpoint():
    ads.set_auth(auth=AuthType.API_KEY, client_kwargs={"service_endpoint": TEST_URL})
    client = OCIFeatureStoreMixin.init_client(fs_service_endpoint=TEST_URL)
    assert client.base_client.endpoint == f"{TEST_URL}/20230101"
