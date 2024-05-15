#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
aqua.model.constants
~~~~~~~~~~~~~~~~~~~~

This module contains constants/enums used in Aqua Model.
"""
from ads.common.extended_enum import ExtendedEnumMeta


class ModelCustomMetadataFields(str, metaclass=ExtendedEnumMeta):
    ARTIFACT_LOCATION = "artifact_location"
    DEPLOYMENT_CONTAINER = "deployment-container"
    EVALUATION_CONTAINER = "evaluation-container"
    FINETUNE_CONTAINER = "finetune-container"


class ModelTask(str, metaclass=ExtendedEnumMeta):
    TEXT_GENERATION = "text-generation"


class FineTuningMetricCategories(str, metaclass=ExtendedEnumMeta):
    VALIDATION = "validation"
    TRAINING = "training"


class ModelType(str, metaclass=ExtendedEnumMeta):
    FT = "FT"  # Fine Tuned Model
    BASE = "BASE"  # Base model


# TODO: merge metadata key used in create FT
class FineTuningCustomMetadata(str, metaclass=ExtendedEnumMeta):
    FT_SOURCE = "fine_tune_source"
    FT_SOURCE_NAME = "fine_tune_source_name"
    FT_OUTPUT_PATH = "fine_tune_output_path"
    FT_JOB_ID = "fine_tune_job_id"
    FT_JOB_RUN_ID = "fine_tune_jobrun_id"
    TRAINING_METRICS_FINAL = "train_metrics_final"
    VALIDATION_METRICS_FINAL = "val_metrics_final"
    TRAINING_METRICS_EPOCH = "train_metrics_epoch"
    VALIDATION_METRICS_EPOCH = "val_metrics_epoch"
