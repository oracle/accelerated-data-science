#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import json
from importlib import reload
from tornado.web import Application
from tornado.testing import AsyncHTTPTestCase

import ads.config
import ads.aqua
from ads.aqua.utils import AQUA_GA_LIST
from ads.aqua.extension.common_handler import CompatibilityCheckHandler


class TestDataset:
    SERVICE_COMPARTMENT_ID = "ocid1.compartment.oc1..<OCID>"


class TestYourHandler(AsyncHTTPTestCase):
    def get_app(self):
        return Application([(r"/hello", CompatibilityCheckHandler)])

    def setUp(self):
        super().setUp()
        os.environ["ODSC_MODEL_COMPARTMENT_OCID"] = TestDataset.SERVICE_COMPARTMENT_ID
        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.extension.common_handler)

    def tearDown(self):
        super().tearDown()
        os.environ.pop("ODSC_MODEL_COMPARTMENT_OCID", None)
        reload(ads.config)
        reload(ads.aqua)
        reload(ads.aqua.extension.common_handler)

    def test_get_ok(self):
        response = self.fetch("/hello", method="GET")
        assert json.loads(response.body)["status"] == "ok"

    def test_get_compatible_status(self):
        os.environ["ODSC_MODEL_COMPARTMENT_OCID"] = ""
        os.environ["CONDA_BUCKET_NS"] = AQUA_GA_LIST[0]
        reload(ads.common)
        reload(ads.aqua)
        reload(ads.aqua.extension.common_handler)
        response = self.fetch("/hello", method="GET")
        assert json.loads(response.body)["status"] == "compatible"

    def test_raise_not_compatible_error(self):
        os.environ["ODSC_MODEL_COMPARTMENT_OCID"] = ""
        os.environ["CONDA_BUCKET_NS"] = "test-namespace"
        reload(ads.common)
        reload(ads.aqua)
        reload(ads.aqua.extension.common_handler)
        response = self.fetch("/hello", method="GET")
        assert (
            json.loads(response.body)["message"]
            == "Authorization Failed: The resource you're looking for isn't accessible."
        )
