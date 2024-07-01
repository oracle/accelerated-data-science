#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""
aqua.evaluation.errors
~~~~~~~~~~~~~~

This module contains errors in Aqua Evaluation.
"""

from ads.common.extended_enum import ExtendedEnumMeta


class EvaluationJobExitCode(str, metaclass=ExtendedEnumMeta):
    SUCCESS = 0
    COMMON_ERROR = 1

    # Configuration-related issues 10-19
    INVALID_EVALUATION_CONFIG = 10
    EVALUATION_CONFIG_NOT_PROVIDED = 11
    INVALID_OUTPUT_DIR = 12
    INVALID_INPUT_DATASET_PATH = 13
    INVALID_EVALUATION_ID = 14
    INVALID_TARGET_EVALUATION_ID = 15
    INVALID_EVALUATION_CONFIG_VALIDATION = 16

    # Evaluation process issues 20-39
    OUTPUT_DIR_NOT_FOUND = 20
    INVALID_INPUT_DATASET = 21
    INPUT_DATA_NOT_FOUND = 22
    EVALUATION_ID_NOT_FOUND = 23
    EVALUATION_ALREADY_PERFORMED = 24
    EVALUATION_TARGET_NOT_FOUND = 25
    NO_SUCCESS_INFERENCE_RESULT = 26
    COMPUTE_EVALUATION_ERROR = 27
    EVALUATION_REPORT_ERROR = 28
    MODEL_INFERENCE_WRONG_RESPONSE_FORMAT = 29
    UNSUPPORTED_METRICS = 30
    METRIC_CALCULATION_FAILURE = 31
    EVALUATION_MODEL_CATALOG_RECORD_CREATION_FAILED = 32


EVALUATION_JOB_EXIT_CODE_MESSAGE = {
    EvaluationJobExitCode.SUCCESS: "Success",
    EvaluationJobExitCode.COMMON_ERROR: "An error occurred during the evaluation, please check the log for more information.",
    EvaluationJobExitCode.INVALID_EVALUATION_CONFIG: "The provided evaluation configuration was not in the correct format, supported formats are YAML or JSON.",
    EvaluationJobExitCode.EVALUATION_CONFIG_NOT_PROVIDED: "The evaluation config was not provided.",
    EvaluationJobExitCode.INVALID_OUTPUT_DIR: "The specified output directory path is invalid.",
    EvaluationJobExitCode.INVALID_INPUT_DATASET_PATH: "Dataset path is invalid.",
    EvaluationJobExitCode.INVALID_EVALUATION_ID: "Evaluation ID was not found in the Model Catalog.",
    EvaluationJobExitCode.INVALID_TARGET_EVALUATION_ID: "Target evaluation ID was not found in the Model Deployment.",
    EvaluationJobExitCode.INVALID_EVALUATION_CONFIG_VALIDATION: "Validation errors in the evaluation config.",
    EvaluationJobExitCode.OUTPUT_DIR_NOT_FOUND: "Destination folder does not exist or cannot be used for writing, verify the folder's existence and permissions.",
    EvaluationJobExitCode.INVALID_INPUT_DATASET: "Input dataset is in an invalid format, ensure the dataset is in jsonl format and that includes the required columns: 'prompt', 'completion' (optional 'category').",
    EvaluationJobExitCode.INPUT_DATA_NOT_FOUND: "Input data file does not exist or cannot be use for reading, verify the file's existence and permissions.",
    EvaluationJobExitCode.EVALUATION_ID_NOT_FOUND: "Evaluation ID does not match any resource in the Model Catalog, or access may be blocked by policies.",
    EvaluationJobExitCode.EVALUATION_ALREADY_PERFORMED: "Evaluation already has an attached artifact, indicating that the evaluation has already been performed.",
    EvaluationJobExitCode.EVALUATION_TARGET_NOT_FOUND: "Target evaluation ID does not match any resources in Model Deployment.",
    EvaluationJobExitCode.NO_SUCCESS_INFERENCE_RESULT: "Inference process completed without producing expected outcome, verify the model parameters and config.",
    EvaluationJobExitCode.COMPUTE_EVALUATION_ERROR: "Evaluation process encountered an issue while calculating metrics.",
    EvaluationJobExitCode.EVALUATION_REPORT_ERROR: "Failed to save the evaluation report due to an error. Ensure the evaluation model is currently active and the specified path for the output report is valid and accessible. Verify these conditions and reinitiate the evaluation process.",
    EvaluationJobExitCode.MODEL_INFERENCE_WRONG_RESPONSE_FORMAT: "Evaluation encountered unsupported, or unexpected model output, verify the target evaluation model is compatible and produces the correct format.",
    EvaluationJobExitCode.UNSUPPORTED_METRICS: "None of the provided metrics are supported by the framework.",
    EvaluationJobExitCode.METRIC_CALCULATION_FAILURE: "All attempted metric calculations were unsuccessful. Please review the metric configurations and input data.",
    EvaluationJobExitCode.EVALUATION_MODEL_CATALOG_RECORD_CREATION_FAILED: (
        "Failed to create a Model Catalog record for the evaluation. "
        "This could be due to missing required permissions. "
        "Please check the log for more information."
    ),
}
