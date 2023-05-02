#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.extended_enum import ExtendedEnum

DEFAULT_OCI_CONFIG_FILE = "~/.oci/config"
DEFAULT_PROFILE = "DEFAULT"
DEFAULT_CONDA_PACK_FOLDER = "~/conda"
DEFAULT_MODEL_FOLDER = "~/.ads_ops/models"
CONDA_PACK_OS_PREFIX_FORMAT = "oci://<bucket>@<namespace>/<prefix>"
DEFAULT_ADS_CONFIG_FOLDER = "~/.ads_ops"
OPS_IMAGE_BASE = "ads-operators-base"
ML_JOB_IMAGE = "ml-job"
ML_JOB_GPU_IMAGE = "ml-job-gpu"
OPS_IMAGE_GPU_BASE = "ads-operators-gpu-base"
DEFAULT_MANIFEST_VERSION = "1.0"
ADS_CONFIG_FILE_NAME = "config.ini"
ADS_JOBS_CONFIG_FILE_NAME = "ml_job_config.ini"
ADS_DATAFLOW_CONFIG_FILE_NAME = "dataflow_config.ini"
ADS_ML_PIPELINE_CONFIG_FILE_NAME = "ml_pipeline.ini"
ADS_LOCAL_BACKEND_CONFIG_FILE_NAME = "local_backend.ini"
ADS_MODEL_DEPLOYMENT_CONFIG_FILE_NAME = "model_deployment_config.ini"
DEFAULT_IMAGE_HOME_DIR = "/home/datascience"
DEFAULT_IMAGE_SCRIPT_DIR = "/etc/datascience"
DEFAULT_IMAGE_CONDA_DIR = "/opt/conda/envs"
DEFAULT_NOTEBOOK_SESSION_SPARK_CONF_DIR = "/home/datascience/spark_conf_dir"
DEFAULT_NOTEBOOK_SESSION_CONDA_DIR = "/home/datascience/conda"
DEFAULT_SPECIFICATION_FILE_NAME = "oci-datascience-template.yaml"
DEFAULT_MODEL_DEPLOYMENT_FOLDER = "/opt/ds/model/deployed_model/"


class RUNTIME_TYPE(ExtendedEnum):
    PYTHON = "python"
    CONTAINER = "container"
    NOTEBOOK = "notebook"
    GITPYTHON = "gitPython"
    OPERATOR = "operator"
    SCRIPT = "script"
    DATAFLOW = "dataFlow"
    DATAFLOWNOTEBOOK = "dataFlowNotebook"
    CONDA = "conda"


class RESOURCE_TYPE(ExtendedEnum):
    JOB = "job"
    DATAFLOW = "dataflow"
    PIPELINE = "pipeline"
    MODEL_DEPLOYMENT = "deployment"


class BACKEND_NAME(ExtendedEnum):
    JOB = "job"
    DATAFLOW = "dataflow"
    PIPELINE = "pipeline"
    MODEL_DEPLOYMENT = "deployment"
    LOCAL = "local"
