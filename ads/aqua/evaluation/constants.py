#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
aqua.evaluation.const
~~~~~~~~~~~~~~

This module contains constants/enums used in Aqua Evaluation.
"""

from oci.data_science.models import JobRun

from ads.common.extended_enum import ExtendedEnum

EVAL_TERMINATION_STATE = [
    JobRun.LIFECYCLE_STATE_SUCCEEDED,
    JobRun.LIFECYCLE_STATE_FAILED,
]


class EvaluationCustomMetadata(ExtendedEnum):
    EVALUATION_SOURCE = "evaluation_source"
    EVALUATION_JOB_ID = "evaluation_job_id"
    EVALUATION_JOB_RUN_ID = "evaluation_job_run_id"
    EVALUATION_OUTPUT_PATH = "evaluation_output_path"
    EVALUATION_SOURCE_NAME = "evaluation_source_name"
    EVALUATION_ERROR = "aqua_evaluate_error"


class EvaluationConfig(ExtendedEnum):
    PARAMS = "model_params"


class EvaluationReportJson(ExtendedEnum):
    """Contains evaluation report.json fields name."""

    METRIC_SUMMARY_RESULT = "metric_summary_result"
    METRIC_RESULT = "metric_results"
    MODEL_PARAMS = "model_params"
    MODEL_DETAILS = "model_details"
    DATA = "data"
    DATASET = "dataset"


class EvaluationMetricResult(ExtendedEnum):
    """Contains metric result's fields name in report.json."""

    SHORT_NAME = "key"
    NAME = "name"
    DESCRIPTION = "description"
    SUMMARY_DATA = "summary_data"
    DATA = "data"
