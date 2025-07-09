#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
aqua.model.constants
~~~~~~~~~~~~~~~~~~~~

This module contains constants/enums used in Aqua Model.
"""

from ads.common.extended_enum import ExtendedEnum


class ModelCustomMetadataFields(ExtendedEnum):
    ARTIFACT_LOCATION = "artifact_location"
    DEPLOYMENT_CONTAINER = "deployment-container"
    EVALUATION_CONTAINER = "evaluation-container"
    FINETUNE_CONTAINER = "finetune-container"
    DEPLOYMENT_CONTAINER_URI = "deployment-container-uri"
    MULTIMODEL_GROUP_COUNT = "model_group_count"
    MULTIMODEL_METADATA = "multi_model_metadata"


class ModelTask(ExtendedEnum):
    TEXT_GENERATION = "text-generation"
    IMAGE_TEXT_TO_TEXT = "image-text-to-text"
    IMAGE_TO_TEXT = "image-to-text"
    TIME_SERIES_FORECASTING = "time-series-forecasting"


class FineTuningMetricCategories(ExtendedEnum):
    VALIDATION = "validation"
    TRAINING = "training"


class ModelType(ExtendedEnum):
    FT = "FT"  # Fine Tuned Model
    BASE = "BASE"  # Base model
    MULTIMODEL = "MULTIMODEL"


# TODO: merge metadata key used in create FT
class FineTuningCustomMetadata(ExtendedEnum):
    FT_SOURCE = "fine_tune_source"
    FT_SOURCE_NAME = "fine_tune_source_name"
    FT_OUTPUT_PATH = "fine_tune_output_path"
    FT_JOB_ID = "fine_tune_job_id"
    FT_JOB_RUN_ID = "fine_tune_jobrun_id"
    TRAINING_METRICS_FINAL = "train_metrics_final"
    VALIDATION_METRICS_FINAL = "val_metrics_final"
    TRAINING_METRICS_EPOCH = "train_metrics_epoch"
    VALIDATION_METRICS_EPOCH = "val_metrics_epoch"


class AquaModelMetadataKeys(ExtendedEnum):
    FINE_TUNING_CONFIGURATION = "FineTuneConfiguration"
    DEPLOYMENT_CONFIGURATION = "DeploymentConfiguration"
    README = "Readme"
    LICENSE = "License"
    REPORTS = "Reports"
