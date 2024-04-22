#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.extended_enum import ExtendedEnumMeta

# Env variable representing the operator input arguments.
# This variable is used when operator run on the OCI resources.
ENV_OPERATOR_ARGS = "ENV_OPERATOR_ARGS"

OPERATOR_BASE_IMAGE = "ads-operator-base"
OPERATOR_BASE_GPU_IMAGE = "ads-operator-gpu-base"
OPERATOR_BASE_DOCKER_FILE = "Dockerfile"
OPERATOR_BASE_DOCKER_GPU_FILE = "Dockerfile.gpu"

OPERATOR_BACKEND_SECTION_NAME = "backend"

class PACK_TYPE(str, metaclass=ExtendedEnumMeta):
    SERVICE = "service"
    CUSTOM = "published"


class ARCH_TYPE(str, metaclass=ExtendedEnumMeta):
    CPU = "cpu"
    GPU = "gpu"
