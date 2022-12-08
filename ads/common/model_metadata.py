#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module created for the back compatability.
The original `model_metadata` was moved to the `ads.model` package.
"""

import warnings

warnings.warn(
    (
        "The `ads.common.model_metadata` is deprecated in `oracle-ads 2.6.8` "
        "and will be removed in future release. "
        "Use the `ads.model.model_metadata` instead."
    ),
    DeprecationWarning,
    stacklevel=2,
)

from ads.model.model_metadata import *
