#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from dataclasses import dataclass

from ads.common.extended_enum import ExtendedEnumMeta
from ads.common.serializer import DataClassSerializable


@dataclass(repr=False)
class AquaResourceIdentifier(DataClassSerializable):
    id: str = ""
    name: str = ""
    url: str = ""


class Resource(str, metaclass=ExtendedEnumMeta):
    JOB = "jobs"
    JOBRUN = "jobruns"
    MODEL = "models"
    MODEL_DEPLOYMENT = "modeldeployments"
    MODEL_VERSION_SET = "model-version-sets"


class DataScienceResource(str, metaclass=ExtendedEnumMeta):
    MODEL_DEPLOYMENT = "datasciencemodeldeployment"
    MODEL = "datasciencemodel"


class Tags(str, metaclass=ExtendedEnumMeta):
    TASK = "task"
    LICENSE = "license"
    ORGANIZATION = "organization"
    AQUA_TAG = "OCI_AQUA"
    AQUA_SERVICE_MODEL_TAG = "aqua_service_model"
    AQUA_FINE_TUNED_MODEL_TAG = "aqua_fine_tuned_model"
    AQUA_MODEL_NAME_TAG = "aqua_model_name"
    AQUA_EVALUATION = "aqua_evaluation"
    AQUA_FINE_TUNING = "aqua_finetuning"
    READY_TO_FINE_TUNE = "ready_to_fine_tune"
    READY_TO_IMPORT = "ready_to_import"
    BASE_MODEL_CUSTOM = "aqua_custom_base_model"


class InferenceContainerType(str, metaclass=ExtendedEnumMeta):
    CONTAINER_TYPE_VLLM = "vllm"
    CONTAINER_TYPE_TGI = "tgi"


class InferenceContainerTypeKey(str, metaclass=ExtendedEnumMeta):
    AQUA_VLLM_CONTAINER_KEY = "odsc-vllm-serving"
    AQUA_TGI_CONTAINER_KEY = "odsc-tgi-serving"


class InferenceContainerParamType(str, metaclass=ExtendedEnumMeta):
    PARAM_TYPE_VLLM = "VLLM_PARAMS"
    PARAM_TYPE_TGI = "TGI_PARAMS"
