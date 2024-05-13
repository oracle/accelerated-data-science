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

from ads.common.extended_enum import Enum

EVAL_TERMINATION_STATE = [
    JobRun.LIFECYCLE_STATE_SUCCEEDED,
    JobRun.LIFECYCLE_STATE_FAILED,
]


class EvaluationCustomMetadata(Enum):
    EVALUATION_SOURCE = "evaluation_source"
    EVALUATION_JOB_ID = "evaluation_job_id"
    EVALUATION_JOB_RUN_ID = "evaluation_job_run_id"
    EVALUATION_OUTPUT_PATH = "evaluation_output_path"
    EVALUATION_SOURCE_NAME = "evaluation_source_name"
    EVALUATION_ERROR = "aqua_evaluate_error"


class EvaluationModelTags(Enum):
    AQUA_EVALUATION = "aqua_evaluation"


class EvaluationJobTags(Enum):
    AQUA_EVALUATION = "aqua_evaluation"
    EVALUATION_MODEL_ID = "evaluation_model_id"


class EvaluationUploadStatus(Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


class RqsAdditionalDetails:
    METADATA = "metadata"
    CREATED_BY = "createdBy"
    DESCRIPTION = "description"
    MODEL_VERSION_SET_ID = "modelVersionSetId"
    MODEL_VERSION_SET_NAME = "modelVersionSetName"
    PROJECT_ID = "projectId"
    VERSION_LABEL = "versionLabel"


class EvaluationConfig:
    PARAMS = "model_params"
