#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from dataclasses import dataclass, field
from typing import Dict

from ads.common.serializer import DataClassSerializable
from ads.model.runtime.env_info import InferenceEnvInfo


@dataclass(repr=False)
class ModelDeploymentDetails(DataClassSerializable):
    """ModelDeploymentDetails class."""

    inference_conda_env: InferenceEnvInfo = field(default_factory=InferenceEnvInfo)

    @staticmethod
    def _validate_dict(obj_dict: Dict) -> bool:
        """Validate the content in the ditionary format from the yaml file.

        Parameters
        ----------
        obj_dict: (Dict)
            yaml file content to validate.

        Returns
        -------
        bool
            Validation result.
        """
        assert obj_dict and (
            "INFERENCE_CONDA_ENV" in obj_dict
        ), "`model_deployment_details` must have `INFERENCE_CONDA_ENV` field."
        return True
