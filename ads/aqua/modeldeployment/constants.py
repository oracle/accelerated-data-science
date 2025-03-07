#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
aqua.modeldeployment.constants
~~~~~~~~~~~~~~

This module contains constants used in Aqua Model Deployment.
"""

VLLMInferenceRestrictedParams = {
    "--tensor-parallel-size",
    "--port",
    "--host",
    "--served-model-name",
    "--seed",
}
TGIInferenceRestrictedParams = {
    "--port",
    "--hostname",
    "--num-shard",
    "--sharded",
    "--trust-remote-code",
}
