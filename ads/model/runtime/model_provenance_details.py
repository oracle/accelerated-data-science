#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from dataclasses import dataclass, field
from typing import Dict

from ads.common.serializer import DataClassSerializable
from ads.model.runtime.env_info import TrainingEnvInfo
from ads.model.runtime.utils import MODEL_PROVENANCE_SCHEMA_PATH, SchemaValidator


@dataclass(repr=False)
class TrainingCode(DataClassSerializable):
    """TrainingCode class."""

    artifact_directory: str = ""

    @classmethod
    def _validate_dict(cls, obj_dict: Dict) -> bool:
        assert obj_dict and (
            "ARTIFACT_DIRECTORY" in obj_dict
        ), "`training_code` must have `ARTIFACT_DIRECTORY` field."
        return True


@dataclass(repr=False)
class ModelProvenanceDetails(DataClassSerializable):
    """ModelProvenanceDetails class."""

    project_ocid: str = ""
    tenancy_ocid: str = ""
    training_code: TrainingCode = field(default_factory=TrainingCode)
    training_compartment_ocid: str = ""
    training_conda_env: TrainingEnvInfo = field(default_factory=TrainingEnvInfo)
    training_region: str = ""
    training_resource_ocid: str = ""
    user_ocid: str = ""
    vm_image_internal_id: str = ""

    @classmethod
    def _validate_dict(cls, obj_dict: Dict) -> bool:
        """validate the yaml file.

        Parameters
        ----------
        obj_dict: (Dict)
            yaml file content to validate.

        Returns
        -------
        bool
            Validation result.
        """
        validator = SchemaValidator(schema_file_path=MODEL_PROVENANCE_SCHEMA_PATH)
        return validator.validate(obj_dict)
