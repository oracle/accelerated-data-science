#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from dataclasses import dataclass
from os import environ
from typing import Union

from ads.model.base_properties import BaseProperties


@dataclass(repr=False)
class ModelProperties(BaseProperties):
    """Represents properties required to save and deploy model."""

    inference_conda_env: str = None
    inference_python_version: str = None
    training_conda_env: str = None
    training_python_version: str = None
    training_resource_id: str = None
    training_script_path: str = None
    training_id: str = None
    compartment_id: str = None
    project_id: str = None
    bucket_uri: str = None
    remove_existing_artifact: bool = None
    overwrite_existing_artifact: bool = None
    deployment_instance_shape: str = None
    deployment_instance_subnet_id: str = None
    deployment_instance_count: int = None
    deployment_bandwidth_mbps: int = None
    deployment_log_group_id: str = None
    deployment_access_log_id: str = None
    deployment_predict_log_id: str = None
    deployment_memory_in_gbs: Union[float, int] = None
    deployment_ocpus: Union[float, int] = None
    deployment_image: str = None

    def _adjust_with_env(self) -> None:
        """Adjusts env variables. This method is used within `with_env` method."""

        super()._adjust_with_env()
        props_env_map = {
            "project_id": ["PROJECT_OCID"],
            "training_resource_id": ["JOB_RUN_OCID", "NB_SESSION_OCID"],
            "compartment_id": [
                "JOB_RUN_COMPARTMENT_OCID",
                "NB_SESSION_COMPARTMENT_OCID",
            ],
        }
        for key, env_keys in props_env_map.items():
            try:
                value = next(
                    environ.get(env_key)
                    for env_key in env_keys
                    if environ.get(env_key, None) is not None
                )
                self.with_prop(key, value)
            except:
                pass
