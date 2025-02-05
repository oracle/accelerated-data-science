#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from ads.common.extended_enum import ExtendedEnum


class FineTuningDefinedMetadata(ExtendedEnum):
    """Represents the defined metadata keys used in Fine Tuning."""

    VAL_SET_SIZE = "val_set_size"
    TRAINING_DATA = "training_data"


class FineTuningCustomMetadata(ExtendedEnum):
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
