#!/usr/bin/env python
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""This module defines constants used in ads.aqua module."""

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

MAXIMUM_ALLOWED_DATASET_IN_BYTE = 52428800  # 1024 x 1024 x 50 = 50MB
JOB_INFRASTRUCTURE_TYPE_DEFAULT_NETWORKING = "ME_STANDALONE"
NB_SESSION_IDENTIFIER = "NB_SESSION_OCID"
LIFECYCLE_DETAILS_MISSING_JOBRUN = "The asscociated JobRun resource has been deleted."
READY_TO_DEPLOY_STATUS = "ACTIVE"
READY_TO_FINE_TUNE_STATUS = "TRUE"
AQUA_GA_LIST = ["id19sfcrra6z"]
AQUA_MODEL_TYPE_SERVICE = "service"
AQUA_MODEL_TYPE_CUSTOM = "custom"
AQUA_MODEL_ARTIFACT_CONFIG = "config.json"
AQUA_MODEL_ARTIFACT_CONFIG_MODEL_NAME = "_name_or_path"
AQUA_MODEL_ARTIFACT_CONFIG_MODEL_TYPE = "model_type"

TRAINING_METRICS_FINAL = "training_metrics_final"
VALIDATION_METRICS_FINAL = "validation_metrics_final"
TRINING_METRICS = "training_metrics"
VALIDATION_METRICS = "validation_metrics"

SERVICE_MANAGED_CONTAINER_URI_SCHEME = "dsmc://"
SUPPORTED_FILE_FORMATS = ["jsonl"]
MODEL_BY_REFERENCE_OSS_PATH_KEY = "artifact_location"

CONSOLE_LINK_RESOURCE_TYPE_MAPPING = {
    "datasciencemodel": "models",
    "datasciencemodeldeployment": "model-deployments",
    "datasciencemodeldeploymentdev": "model-deployments",
    "datasciencemodeldeploymentint": "model-deployments",
    "datasciencemodeldeploymentpre": "model-deployments",
    "datasciencejob": "jobs",
    "datasciencejobrun": "job-runs",
    "datasciencejobrundev": "job-runs",
    "datasciencejobrunint": "job-runs",
    "datasciencejobrunpre": "job-runs",
    "datasciencemodelversionset": "model-version-sets",
    "datasciencemodelversionsetpre": "model-version-sets",
    "datasciencemodelversionsetint": "model-version-sets",
    "datasciencemodelversionsetdev": "model-version-sets",
}

VLLM_INFERENCE_RESTRICTED_PARAMS = {
    "--port",
    "--host",
    "--served-model-name",
    "--seed",
}
TGI_INFERENCE_RESTRICTED_PARAMS = {
    "--port",
    "--hostname",
    "--num-shard",
    "--sharded",
    "--trust-remote-code",
}
