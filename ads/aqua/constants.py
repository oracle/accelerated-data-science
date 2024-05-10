#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""This module defines constants used in ads.aqua module."""
from enum import Enum

# TODO: import from aqua.constants
UNKNOWN = ""
UNKNOWN_VALUE = ""
READY_TO_IMPORT_STATUS = "TRUE"
UNKNOWN_DICT = {}
README = "README.md"
LICENSE_TXT = "config/LICENSE.txt"
DEPLOYMENT_CONFIG = "deployment_config.json"
COMPARTMENT_MAPPING_KEY = "service-model-compartment"
CONTAINER_INDEX = "container_index.json"
EVALUATION_REPORT_JSON = "report.json"
EVALUATION_REPORT_MD = "report.md"
EVALUATION_REPORT = "report.html"
UNKNOWN_JSON_STR = "{}"
FINE_TUNING_RUNTIME_CONTAINER = "iad.ocir.io/ociodscdev/aqua_ft_cuda121:0.3.17.20"
DEFAULT_FT_BLOCK_STORAGE_SIZE = 750
DEFAULT_FT_REPLICA = 1
DEFAULT_FT_BATCH_SIZE = 1
DEFAULT_FT_VALIDATION_SET_SIZE = 0.1

HF_MODELS = "/home/datascience/conda/pytorch21_p39_gpu_v1/"
MAXIMUM_ALLOWED_DATASET_IN_BYTE = 52428800  # 1024 x 1024 x 50 = 50MB
JOB_INFRASTRUCTURE_TYPE_DEFAULT_NETWORKING = "ME_STANDALONE"
NB_SESSION_IDENTIFIER = "NB_SESSION_OCID"
LIFECYCLE_DETAILS_MISSING_JOBRUN = "The asscociated JobRun resource has been deleted."
READY_TO_DEPLOY_STATUS = "ACTIVE"
READY_TO_FINE_TUNE_STATUS = "TRUE"
AQUA_GA_LIST = ["id19sfcrra6z"]
AQUA_MODEL_TYPE_SERVICE = "service"
AQUA_MODEL_TYPE_CUSTOM = "custom"


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

SERVICE_MANAGED_CONTAINER_URI_SCHEME = "dsmc://"
