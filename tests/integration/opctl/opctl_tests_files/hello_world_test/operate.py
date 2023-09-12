#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from test_mod import test_import
import ads


def operate(spec=None, **kwargs):
    print(f"ADS {ads.__version__}")
    test_import()
    print(kwargs["oci_auth"])
    name = spec["name"] if spec else ""
    print("Hello World", name)
    if "TEST_ENV" in os.environ:
        print(f"Test passing environment variables TEST_ENV={os.environ['TEST_ENV']}")
