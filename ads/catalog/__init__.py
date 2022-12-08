#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import warnings

warnings.warn(
    (
        "The `ads.catalog` is deprecated in `oracle-ads 2.6.9` and will be removed in `oracle-ads 3.0`."
    ),
    DeprecationWarning,
    stacklevel=2,
)

import logging

logger = logging.getLogger(__name__)
