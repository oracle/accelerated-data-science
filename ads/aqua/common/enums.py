#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
aqua.common.enums
~~~~~~~~~~~~~~
This module contains the set of enums used in AQUA.
"""
from ads.common.extended_enum import ExtendedEnumMeta


class Resource(str, metaclass=ExtendedEnumMeta):
    JOB = "jobs"
    MODEL = "models"
    MODEL_DEPLOYMENT = "modeldeployments"
    MODEL_VERSION_SET = "model-version-sets"


class DataScienceResource(str, metaclass=ExtendedEnumMeta):
    MODEL_DEPLOYMENT = "datasciencemodeldeployment"
    MODEL = "datasciencemodel"
