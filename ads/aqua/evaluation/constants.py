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


class EvaluationConfig(str, metaclass=ExtendedEnumMeta):
    PARAMS = "model_params"


class EvaluationReportJson(str, metaclass=ExtendedEnumMeta):
    """Contains evaluation report.json fields name."""

    METRIC_SUMMARY_RESULT = "metric_summary_result"
    METRIC_RESULT = "metric_results"
    MODEL_PARAMS = "model_params"
    MODEL_DETAILS = "model_details"
    DATA = "data"
    DATASET = "dataset"


class EvaluationMetricResult(str, metaclass=ExtendedEnumMeta):
    """Contains metric result's fields name in report.json."""

    SHORT_NAME = "key"
    NAME = "name"
    DESCRIPTION = "description"
    SUMMARY_DATA = "summary_data"
    DATA = "data"
