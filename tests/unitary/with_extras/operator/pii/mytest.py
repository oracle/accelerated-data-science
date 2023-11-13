#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl.operator.lowcode.pii.model.pii import Scrubber

from ads.opctl.operator.common.utils import _load_yaml_from_uri


test_yaml_uri = "/Users/mingkang/workspace/github/accelerated-data-science/tests/unitary/with_extras/operator/pii/test_files/pii_test.yaml"

# config = _load_yaml_from_uri(uri=test_yaml_uri)

# print(config)

import scrubadub
from ads.opctl.operator.lowcode.pii.model.processor import Remover
from ads.opctl.operator.lowcode.pii.model.factory import PiiDetectorFactory
from ads.opctl.operator.lowcode.pii.model.pii import Scrubber

text = """
This is John Doe. My number is (213)275-8452.
"""

scrubber = Scrubber(config=test_yaml_uri)


# scrubber = scrubadub.Scrubber()
# print(scrubber._post_processors)
print(scrubber.scrubber._detectors)
# scrubber.add_detector("phone")
# # remover = Remover()
# # remover._ENTITIES.append("phone")
# # scrubber.add_post_processor(remover)
# scrubber._detectors["phone"].filth_cls.replacement_string = "***"
# print(scrubber._detectors["phone"].filth_cls.replacement_string)
# out = scrubber.clean(text)
print(scrubber.scrubber._post_processors)
out = scrubber.scrubber.clean(text)

print(out)
