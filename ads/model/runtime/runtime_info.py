#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from dataclasses import dataclass, field
from typing import Dict

from ads.common.serializer import DataClassSerializable, SideEffect
from ads.model.runtime.model_deployment_details import ModelDeploymentDetails
from ads.model.runtime.model_provenance_details import ModelProvenanceDetails


@dataclass(repr=False)
class RuntimeInfo(DataClassSerializable):
    """RuntimeInfo class which is the data class represenation of the runtime yaml file."""

    model_artifact_version: str = ""
    model_deployment: ModelDeploymentDetails = field(
        default_factory=ModelDeploymentDetails
    )
    model_provenance: ModelProvenanceDetails = field(
        default_factory=ModelProvenanceDetails
    )

    @classmethod
    def _validate_dict(cls, obj_dict: Dict) -> bool:
        """Validate the runtime info.

        Parameters
        ----------
        obj_dict: (Dict)
            runtime content in dictionary format.

        Returns
        -------
        bool
            the validation result.
        """
        assert (
            "MODEL_ARTIFACT_VERSION" in obj_dict
        ), "runtime.yaml must have `MODEL_ARTIFACT_VERSION` field."
        assert (
            "MODEL_DEPLOYMENT" in obj_dict
        ), "runtime.yaml must have `MODEL_DEPLOYMENT` field."
        assert (
            "MODEL_PROVENANCE" in obj_dict
        ), "runtime.yaml must have `MODEL_PROVENANCE` field."
        return True

    @classmethod
    def from_env(cls) -> "RuntimeInfo":
        """Popolate the RuntimeInfo from environment variables.

        Returns
        -------
        RuntimeInfo
            A RuntimeInfo instance.
        """
        runtime_info = cls(model_artifact_version="3.0")
        return runtime_info

    def save(self, storage_options=None):
        """Save the RuntimeInfo object into runtime.yaml file under the artifact directory.

        Returns
        -------
        None
            Nothing.
        """
        runtime_file_path = os.path.join(
            self.model_provenance.training_code.artifact_directory, "runtime.yaml"
        )
        storage_options = storage_options or {}
        self.to_yaml(
            uri=runtime_file_path,
            side_effect=SideEffect.CONVERT_KEYS_TO_UPPER.value,
            storage_options=storage_options,
        )
