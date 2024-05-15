#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
aqua.evaluation.const
~~~~~~~~~~~~~~

This module contains constants/enums used in Aqua Evaluation.
"""
from oci.data_science.models import JobRun

from ads.common.extended_enum import ExtendedEnumMeta

EVAL_TERMINATION_STATE = [
    JobRun.LIFECYCLE_STATE_SUCCEEDED,
    JobRun.LIFECYCLE_STATE_FAILED,
]


class EvaluationCustomMetadata(str, metaclass=ExtendedEnumMeta):
    EVALUATION_SOURCE = "evaluation_source"
    EVALUATION_JOB_ID = "evaluation_job_id"
    EVALUATION_JOB_RUN_ID = "evaluation_job_run_id"
    EVALUATION_OUTPUT_PATH = "evaluation_output_path"
    EVALUATION_SOURCE_NAME = "evaluation_source_name"
    EVALUATION_ERROR = "aqua_evaluate_error"


class EvaluationModelTags(str, metaclass=ExtendedEnumMeta):
    AQUA_EVALUATION = "aqua_evaluation"


class EvaluationJobTags(str, metaclass=ExtendedEnumMeta):
    AQUA_EVALUATION = "aqua_evaluation"
    EVALUATION_MODEL_ID = "evaluation_model_id"


class EvaluationUploadStatus(str, metaclass=ExtendedEnumMeta):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


class RqsAdditionalDetails(str, metaclass=ExtendedEnumMeta):
    METADATA = "metadata"
    CREATED_BY = "createdBy"
    DESCRIPTION = "description"
    MODEL_VERSION_SET_ID = "modelVersionSetId"
    MODEL_VERSION_SET_NAME = "modelVersionSetName"
    PROJECT_ID = "projectId"
    VERSION_LABEL = "versionLabel"


class EvaluationConfig(str, metaclass=ExtendedEnumMeta):
    PARAMS = "model_params"
