#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import ads
from ads.feature_store.mixin.oci_feature_store import OCIFeatureStoreMixin
from ads.common.auth import AuthType

TEST_URL_1 = "https://test1.com"
TEST_URL_2 = "https://test2.com"
TEST_URL_3 = "https://test3.com"
TEST_URL_4 = "https://test4.com"


def test_global_service_endpoint():
    ads.set_auth(auth=AuthType.API_KEY, client_kwargs={"service_endpoint": TEST_URL_1})
    client = OCIFeatureStoreMixin.init_client()
    assert client.base_client.endpoint == f"{TEST_URL_1}/20230101"


def test_global_service_and_fs_endpoints():
    ads.set_auth(
        auth=AuthType.API_KEY,
        client_kwargs={
            "fs_service_endpoint": TEST_URL_1,
            "service_endpoint": TEST_URL_2,
        },
    )
    client = OCIFeatureStoreMixin.init_client()
    assert client.base_client.endpoint == f"{TEST_URL_1}/20230101"


def test_override_service_endpoint():
    ads.set_auth(auth=AuthType.API_KEY)
    client = OCIFeatureStoreMixin.init_client(service_endpoint=TEST_URL_1)
    assert client.base_client.endpoint == f"{TEST_URL_1}/20230101"


def test_override_service_and_fs_endpoints():
    ads.set_auth(auth=AuthType.API_KEY)
    client = OCIFeatureStoreMixin.init_client(
        service_endpoint=TEST_URL_1, fs_service_endpoint=TEST_URL_2
    )
    assert client.base_client.endpoint == f"{TEST_URL_2}/20230101"


def test_override_service_and_fs_endpoints_with_global_service_and_fs_endpoints():
    ads.set_auth(
        auth=AuthType.API_KEY,
        client_kwargs={
            "fs_service_endpoint": TEST_URL_3,
            "service_endpoint": TEST_URL_4,
        },
    )
    client = OCIFeatureStoreMixin.init_client(
        service_endpoint=TEST_URL_1, fs_service_endpoint=TEST_URL_2
    )
    assert client.base_client.endpoint == f"{TEST_URL_2}/20230101"
