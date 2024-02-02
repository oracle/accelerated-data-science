#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import unittest
from collections import namedtuple
from dataclasses import asdict
from unittest.mock import ANY, MagicMock, patch

from ads.aqua.model import AquaModelApp, AquaModelSummary


class TestDataset:
    Response = namedtuple("Response", ["data", "status"])
    DataList = namedtuple("DataList", ["objects"])

    resource_summary_objects = []

    COMPARTMENT_ID = "ocid1.compartment.oc1..<OCID>"
    MOCK_ICON = "data:image/svg+xml;base64,########"


class TestAquaModel(unittest.TestCase):
    """Contains unittests for AquaModelApp."""

    def test_list(self):
        """Tests list models succesfully."""
        pass

    def test_list_failed(self):
        """Tests list models succesfully."""
        pass
