#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import oci
from ads.common.auth import default_signer


class AquaApp:
    """Base Aqua App to contain common components."""

    def __init__(self) -> None:
        self.client = oci.data_science.DataScienceClient(**default_signer())
