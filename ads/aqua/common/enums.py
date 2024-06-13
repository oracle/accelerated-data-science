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


class DataScienceResource(str, metaclass=ExtendedEnumMeta):
    MODEL_DEPLOYMENT = "datasciencemodeldeployment"
    MODEL = "datasciencemodel"


class Resource(str, metaclass=ExtendedEnumMeta):
    JOB = "jobs"
    JOBRUN = "jobruns"
    MODEL = "models"
    MODEL_DEPLOYMENT = "modeldeployments"
    MODEL_VERSION_SET = "model-version-sets"


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
    AQUA_EVALUATION_MODEL_ID = "evaluation_model_id"


class InferenceContainerType(str, metaclass=ExtendedEnumMeta):
    CONTAINER_TYPE_VLLM = "vllm"
    CONTAINER_TYPE_TGI = "tgi"


class InferenceContainerTypeFamily(str, metaclass=ExtendedEnumMeta):
    AQUA_VLLM_CONTAINER_FAMILY = "odsc-vllm-serving"
    AQUA_TGI_CONTAINER_FAMILY = "odsc-tgi-serving"


class InferenceContainerParamType(str, metaclass=ExtendedEnumMeta):
    PARAM_TYPE_VLLM = "VLLM_PARAMS"
    PARAM_TYPE_TGI = "TGI_PARAMS"


class HuggingFaceTags(str, metaclass=ExtendedEnumMeta):
    TEXT_GENERATION_INFERENCE = "text-generation-inference"


class RqsAdditionalDetails(str, metaclass=ExtendedEnumMeta):
    METADATA = "metadata"
    CREATED_BY = "createdBy"
    DESCRIPTION = "description"
    MODEL_VERSION_SET_ID = "modelVersionSetId"
    MODEL_VERSION_SET_NAME = "modelVersionSetName"
    PROJECT_ID = "projectId"
    VERSION_LABEL = "versionLabel"
