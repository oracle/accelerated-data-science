#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import os

import pytest

from ads.opctl.operator.common.utils import _load_yaml_from_uri
from ads.opctl.operator.lowcode.pii.model.pii import PiiScrubber
from ads.opctl.operator.lowcode.pii.operator_config import PiiOperatorConfig


class TestPiiScrubber:
    test_yaml_uri = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_files", "pii_test.yaml"
    )
    operator_config = PiiOperatorConfig.from_yaml(uri=test_yaml_uri)
    config_dict = _load_yaml_from_uri(uri=test_yaml_uri)

    name_entity = "John Doe"
    phone_entity = "(800) 223-1711"
    text = f"""
    This is {name_entity}. My number is {phone_entity}.
    """

    @pytest.mark.parametrize(
        "config",
        [
            test_yaml_uri,
            operator_config,
            config_dict,
        ],
    )
    def test_init(self, config):
        pii_scrubber = PiiScrubber(config=config)

        assert isinstance(pii_scrubber.detector_spec, list)
        assert len(pii_scrubber.detector_spec) == 2
        assert pii_scrubber.detector_spec[0]["name"] == "default.phone"

        assert len(pii_scrubber.scrubber._detectors) == 0

    def test_config_scrubber(self):
        scrubber = PiiScrubber(config=self.test_yaml_uri).config_scrubber()

        assert len(scrubber._detectors) == 2
        assert len(scrubber._post_processors) == 1

        processed_text = scrubber.clean(self.text)

        assert self.name_entity not in processed_text
        assert self.phone_entity not in processed_text
