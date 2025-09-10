#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""This module defines constants used in ads.aqua module."""

UNKNOWN_VALUE = ""
READY_TO_IMPORT_STATUS = "TRUE"
UNKNOWN_DICT = {}
DEPLOYMENT_CONFIG = "deployment_config.json"
FINE_TUNING_CONFIG = "ft_config.json"
README = "README.md"
LICENSE = "LICENSE.txt"
AQUA_MODEL_TOKENIZER_CONFIG = "tokenizer_config.json"
COMPARTMENT_MAPPING_KEY = "service-model-compartment"
CONTAINER_INDEX = "container_index.json"
EVALUATION_REPORT_JSON = "report.json"
EVALUATION_REPORT_MD = "report.md"
EVALUATION_REPORT = "report.html"
UNKNOWN_JSON_STR = "{}"
UNKNOWN_JSON_LIST = "[]"
FINE_TUNING_RUNTIME_CONTAINER = "iad.ocir.io/ociodscdev/aqua_ft_cuda121:0.3.17.20"
DEFAULT_FT_BLOCK_STORAGE_SIZE = 750
DEFAULT_FT_REPLICA = 1
DEFAULT_FT_BATCH_SIZE = 1
DEFAULT_FT_VALIDATION_SET_SIZE = 0.1
MAXIMUM_ALLOWED_DATASET_IN_BYTE = 52428800  # 1024 x 1024 x 50 = 50MB
JOB_INFRASTRUCTURE_TYPE_DEFAULT_NETWORKING = "ME_STANDALONE"
NB_SESSION_IDENTIFIER = "NB_SESSION_OCID"
LIFECYCLE_DETAILS_MISSING_JOBRUN = "The associated JobRun resource has been deleted."
READY_TO_DEPLOY_STATUS = "ACTIVE"
READY_TO_FINE_TUNE_STATUS = "TRUE"
PRIVATE_ENDPOINT_TYPE = "MODEL_DEPLOYMENT"
AQUA_GA_LIST = ["id19sfcrra6z"]
AQUA_MULTI_MODEL_CONFIG = "MULTI_MODEL_CONFIG"
AQUA_MODEL_TYPE_SERVICE = "service"
AQUA_MODEL_TYPE_CUSTOM = "custom"
AQUA_MODEL_TYPE_MULTI = "multi_model"
AQUA_MODEL_ARTIFACT_CONFIG = "config.json"
AQUA_MODEL_ARTIFACT_CONFIG_MODEL_NAME = "_name_or_path"
AQUA_MODEL_ARTIFACT_CONFIG_MODEL_TYPE = "model_type"
AQUA_MODEL_ARTIFACT_FILE = "model_file"
HF_METADATA_FOLDER = ".cache/"
HF_LOGIN_DEFAULT_TIMEOUT = 2
MODEL_NAME_DELIMITER = ";"
AQUA_TROUBLESHOOTING_LINK = "https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/ai-quick-actions/troubleshooting-tips.md"
MODEL_FILE_DESCRIPTION_VERSION = "1.0"
MODEL_FILE_DESCRIPTION_TYPE = "modelOSSReferenceDescription"
AQUA_FINE_TUNE_MODEL_VERSION = "v2"
INCLUDE_BASE_MODEL = 1

TRAINING_METRICS_FINAL = "training_metrics_final"
VALIDATION_METRICS_FINAL = "validation_metrics_final"
TRINING_METRICS = "training_metrics"
VALIDATION_METRICS = "validation_metrics"

SERVICE_MANAGED_CONTAINER_URI_SCHEME = "dsmc://"
SUPPORTED_FILE_FORMATS = ["jsonl"]
MODEL_BY_REFERENCE_OSS_PATH_KEY = "artifact_location"

AQUA_CHAT_TEMPLATE_METADATA_KEY = "chat_template"
UNKNOWN_ENUM_VALUE = "UNKNOWN_ENUM_VALUE"
MODEL_GROUP = "MODEL_GROUP"
SINGLE_MODEL_FLEX = "SINGLE_MODEL_FLEX"

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
LLAMA_CPP_INFERENCE_RESTRICTED_PARAMS = {
    "--port",
    "--host",
}
TEI_CONTAINER_DEFAULT_HOST = "8080"

OCI_OPERATION_FAILURES = {
    "list_model_deployments": "Unable to list model deployments. See tips for troubleshooting: ",
    "list_models": "Unable to list models. See tips for troubleshooting: ",
    "get_namespace": "Unable to access specified Object Storage Bucket. See tips for troubleshooting: ",
    "list_log_groups": "Unable to access logs. See tips for troubleshooting: ",
    "list_buckets": "Unable to list Object Storage Bucket. See tips for troubleshooting: ",
    "put_object": "Unable to access or find Object Storage Bucket. See tips for troubleshooting: ",
    "list_model_version_sets": "Unable to create or fetch model version set. See tips for troubleshooting:",
    "update_model": "Unable to update model. See tips for troubleshooting: ",
    "list_data_science_private_endpoints": "Unable to access private endpoint. See tips for troubleshooting:  ",
    "create_model": "Unable to register model. See tips for troubleshooting: ",
    "create_deployment": "Unable to create deployment. See tips for troubleshooting: ",
    "create_model_version_sets": "Unable to create model version set. See tips for troubleshooting: ",
    "create_job": "Unable to create job. See tips for troubleshooting: ",
    "create_job_run": "Unable to create job run. See tips for troubleshooting: ",
}

STATUS_CODE_MESSAGES = {
    "400": "Could not process your request due to invalid input.",
    "403": "We're having trouble processing your request with the information provided.",
    "404": "Authorization Failed: The resource you're looking for isn't accessible.",
    "408": "Server is taking too long to respond, please try again.",
}
