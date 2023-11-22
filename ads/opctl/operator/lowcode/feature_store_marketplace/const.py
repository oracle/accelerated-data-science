#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.extended_enum import ExtendedEnumMeta


class DBType(str, metaclass=ExtendedEnumMeta):
    """Supported databases for feature store"""

    MySQL = "MYSQL"
    ATP = "ATP"
