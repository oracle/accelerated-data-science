#!/usr/bin/env python
# -*- coding: utf-8 -*--
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest

from ads.aqua.data import AquaResourceIdentifier


@pytest.fixture
def valid_data():
    return {
        "id": "ocid.datasciencemodel.oc1..xxxx",
        "name": "Resource",
        "region": "myregion",
    }


@pytest.fixture
def invalid_data():
    return {
        "random_attr": "123",
    }


def test_from_data_with_invalid_data(invalid_data):
    instance = AquaResourceIdentifier.from_data(data=invalid_data)
    assert instance.id == ""
    assert instance.name == ""
    assert instance.url == ""


def test_create_instance_with_post_init(valid_data):
    instance = AquaResourceIdentifier(**valid_data)
    assert instance.id == valid_data["id"]
    assert instance.name == valid_data["name"]
    assert (
        instance.url
        == "https://cloud.oracle.com/data-science/models/ocid.datasciencemodel.oc1..xxxx?region=myregion"
    )


def test_from_data_method(valid_data):
    instance = AquaResourceIdentifier.from_data(valid_data)
    assert instance.id == valid_data["id"]
    assert instance.name == valid_data["name"]
