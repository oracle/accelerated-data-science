#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from ads.llm.langchain.plugins import GenerativeAI
from ads.llm.langchain.plugins import ModelDeploymentTGI

# from ads.llm.langchain.plugins import ModelDeploymentVLLM


__all__ = [
    "GenerativeAI",
    "ModelDeploymentTGI",
    # "ModelDeploymentVLLM",
]
