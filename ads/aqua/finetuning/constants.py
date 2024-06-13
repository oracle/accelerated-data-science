#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.extended_enum import ExtendedEnumMeta


class FineTuneCustomMetadata(str, metaclass=ExtendedEnumMeta):
    FINE_TUNE_SOURCE = "fine_tune_source"
    FINE_TUNE_SOURCE_NAME = "fine_tune_source_name"
    FINE_TUNE_OUTPUT_PATH = "fine_tune_output_path"
    FINE_TUNE_JOB_ID = "fine_tune_job_id"
    FINE_TUNE_JOB_RUN_ID = "fine_tune_job_run_id"
    SERVICE_MODEL_ARTIFACT_LOCATION = "artifact_location"
    SERVICE_MODEL_DEPLOYMENT_CONTAINER = "deployment-container"
    SERVICE_MODEL_FINE_TUNE_CONTAINER = "finetune-container"
