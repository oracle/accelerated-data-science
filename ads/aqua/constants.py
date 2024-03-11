#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""This module defines constants used in ads.aqua module."""
from enum import Enum

UNKNOWN_VALUE = ""


class RqsAdditionalDetails:
    METADATA = "metadata"
    CREATED_BY = "createdBy"
    DESCRIPTION = "description"
    MODEL_VERSION_SET_ID = "modelVersionSetId"
    MODEL_VERSION_SET_NAME = "modelVersionSetName"
    PROJECT_ID = "projectId"
    VERSION_LABEL = "versionLabel"


class FineTuningDefinedMetadata(Enum):
    """Represents the defined metadata keys used in Fine Tuning."""

    VAL_SET_SIZE = "val_set_size"
    TRAINING_DATA = "training_data"


class FineTuningCustomMetadata(Enum):
    """Represents the custom metadata keys used in Fine Tuning."""

    FT_SOURCE = "fine_tune_source"
    FT_SOURCE_NAME = "fine_tune_source_name"
    FT_OUTPUT_PATH = "fine_tune_output_path"
    FT_JOB_ID = "fine_tune_job_id"
    FT_JOB_RUN_ID = "fine_tune_jobrun_id"
    TRAINING_METRICS_FINAL = "train_metrics_final"
    VALIDATION_METRICS_FINAL = "val_metrics_final"
    TRAINING_METRICS_EPOCH = "train_metrics_epoch"
    VALIDATION_METRICS_EPOCH = "val_metrics_epoch"


TRAINING_METRICS_FINAL = "training_metrics_final"
VALIDATION_METRICS_FINAL = "validation_metrics_final"
TRINING_METRICS = "training_metrics"
VALIDATION_METRICS = "validation_metrics"
