#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.extended_enum import ExtendedEnumMeta


class InferenceContainerType(str, metaclass=ExtendedEnumMeta):
    CONTAINER_TYPE_VLLM = "vllm"
    CONTAINER_TYPE_TGI = "tgi"


class InferenceContainerTypeKey(str, metaclass=ExtendedEnumMeta):
    AQUA_VLLM_CONTAINER_KEY = "odsc-vllm-serving"
    AQUA_TGI_CONTAINER_KEY = "odsc-tgi-serving"


class InferenceContainerParamType(str, metaclass=ExtendedEnumMeta):
    PARAM_TYPE_VLLM = "VLLM_PARAMS"
    PARAM_TYPE_TGI = "TGI_PARAMS"
