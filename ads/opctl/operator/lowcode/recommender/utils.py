#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os


def default_signer(**kwargs):
    os.environ["EXTRA_USER_AGENT_INFO"] = "Recommender-Operator"
    from ads.common.auth import default_signer
    return default_signer(**kwargs)
