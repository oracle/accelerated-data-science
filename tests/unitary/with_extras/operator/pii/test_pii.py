#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import unittest
from unittest.mock import MagicMock, patch

from ads.opctl.operator.lowcode.pii.model.pii import Scrubber
from ads.opctl.operator.lowcode.pii.operator_config import PiiOperatorConfig


class TestScrubber(unittest.TestCase):
    test_yaml_uri = "/Users/mingkang/workspace/github/accelerated-data-science/tests/unitary/with_extras/operator/pii/test_files/pii_test.yaml"
    operator_config = PiiOperatorConfig.from_yaml(uri=test_yaml_uri)
    config_dict = {}

    def test_init_with_yaml_file(self):
        scrubber = Scrubber(config=self.test_yaml_uri)

    def test_init_with_piiOperatorConfig(self):
        scrubber = Scrubber(config=self.operator_config)

    def test_init_with_config_dict(self):
        scrubber = Scrubber(config=self.config_dict)
