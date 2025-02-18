#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
from typing import List, Literal, Optional, Union

from pydantic import Field, model_validator

from ads.aqua.common.errors import AquaValueError
from ads.aqua.config.utils.serializer import Serializable
from ads.aqua.data import AquaResourceIdentifier
from ads.aqua.finetuning.constants import FineTuningRestrictedParams


class AquaFineTuningParams(Serializable):
    """Class for maintaining aqua fine-tuning model parameters"""

    epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    sample_packing: Union[bool, None, Literal["auto"]] = "auto"
    batch_size: Optional[int] = (
        None  # make it batch_size for user, but internally this is micro_batch_size
    )
    sequence_len: Optional[int] = None
    pad_to_sequence_len: Optional[bool] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    lora_target_linear: Optional[bool] = None
    lora_target_modules: Optional[List[str]] = None
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: Optional[float] = None

    class Config:
        extra = "allow"

    def to_dict(self) -> dict:
        return json.loads(super().to_json(exclude_none=True))

    @model_validator(mode="before")
    @classmethod
    def validate_restricted_fields(cls, data: dict):
        # we may want to skip validation if loading data from config files instead of user entered parameters
        validate = data.pop("_validate", True)
        if not (validate and isinstance(data, dict)):
            return data
        restricted_params = [
            param for param in data if param in FineTuningRestrictedParams.values()
        ]
        if restricted_params:
            raise AquaValueError(
                f"Found restricted parameter name: {restricted_params}"
            )
        return data


class AquaFineTuningSummary(Serializable):
    """Represents a summary of Aqua Finetuning job."""

    id: str
    name: str
    console_url: str
    lifecycle_state: str
    lifecycle_details: str
    time_created: str
    tags: dict
    experiment: AquaResourceIdentifier = Field(default_factory=AquaResourceIdentifier)
    source: AquaResourceIdentifier = Field(default_factory=AquaResourceIdentifier)
    job: AquaResourceIdentifier = Field(default_factory=AquaResourceIdentifier)
    parameters: AquaFineTuningParams = Field(default_factory=AquaFineTuningParams)

    class Config:
        extra = "ignore"

    def to_dict(self) -> dict:
        return json.loads(super().to_json(exclude_none=True))


class CreateFineTuningDetails(Serializable):
    """Class to create aqua model fine-tuning instance.

    Properties
    ------
    ft_source_id: str
        The fine tuning source id. Must be model ocid.
    ft_name: str
        The name for fine tuning.
    dataset_path: str
        The dataset path for fine tuning. Could be either a local path from notebook session
        or an object storage path.
    report_path: str
        The report path for fine tuning. Must be an object storage path.
    ft_parameters: dict
        The parameters for fine tuning.
    shape_name: str
        The shape name for fine tuning job infrastructure.
    replica: int
        The replica for fine tuning job runtime.
    validation_set_size: float
        The validation set size for fine tuning job. Must be a float in between [0,1).
    ft_description: (str, optional). Defaults to `None`.
        The description for fine tuning.
    compartment_id: (str, optional). Defaults to `None`.
        The compartment id for fine tuning.
    project_id: (str, optional). Defaults to `None`.
        The project id for fine tuning.
    experiment_id: (str, optional). Defaults to `None`.
        The fine tuning model version set id. If provided,
        fine tuning model will be associated with it.
    experiment_name: (str, optional). Defaults to `None`.
        The fine tuning model version set name. If provided,
        the fine tuning version set with the same name will be used if exists,
        otherwise a new model version set will be created with the name.
    experiment_description: (str, optional). Defaults to `None`.
        The description for fine tuning model version set.
    block_storage_size: (int, optional). Defaults to 256.
        The storage for fine tuning job infrastructure.
    subnet_id: (str, optional). Defaults to `None`.
        The custom egress for fine tuning job.
    log_group_id: (str, optional). Defaults to `None`.
        The log group id for fine tuning job infrastructure.
    log_id: (str, optional). Defaults to `None`.
        The log id for fine tuning job infrastructure.
    watch_logs: (bool, optional). Defaults to `False`.
        The flag to watch the job run logs when a fine-tuning job is created.
    force_overwrite: (bool, optional). Defaults to `False`.
        Whether to force overwrite the existing file in object storage.
    freeform_tags: (dict, optional)
        Freeform tags for the fine-tuning model
    defined_tags: (dict, optional)
        Defined tags for the fine-tuning model
    """

    ft_source_id: str
    ft_name: str
    dataset_path: str
    report_path: str
    ft_parameters: dict
    shape_name: str
    replica: int
    validation_set_size: float
    ft_description: Optional[str] = None
    compartment_id: Optional[str] = None
    project_id: Optional[str] = None
    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None
    experiment_description: Optional[str] = None
    block_storage_size: Optional[int] = None
    subnet_id: Optional[str] = None
    log_id: Optional[str] = None
    log_group_id: Optional[str] = None
    watch_logs: Optional[bool] = False
    force_overwrite: Optional[bool] = False
    freeform_tags: Optional[dict] = None
    defined_tags: Optional[dict] = None

    class Config:
        extra = "ignore"
