#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from dataclasses import dataclass, field
from typing import List, Optional

from ads.aqua.data import AquaJobSummary
from ads.common.serializer import DataClassSerializable


@dataclass(repr=False)
class AquaFineTuningParams(DataClassSerializable):
    epochs: int
    learning_rate: Optional[float] = None
    sample_packing: Optional[bool] = "auto"
    batch_size: Optional[
        int
    ] = None  # make it batch_size for user, but internally this is micro_batch_size
    sequence_len: Optional[int] = None
    pad_to_sequence_len: Optional[bool] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    lora_target_linear: Optional[bool] = None
    lora_target_modules: Optional[List] = None


@dataclass(repr=False)
class AquaFineTuningSummary(AquaJobSummary, DataClassSerializable):
    parameters: AquaFineTuningParams = field(default_factory=AquaFineTuningParams)


@dataclass(repr=False)
class CreateFineTuningDetails(DataClassSerializable):
    """Dataclass to create aqua model fine tuning.

    Fields
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
    force_overwrite: (bool, optional). Defaults to `False`.
        Whether to force overwrite the existing file in object storage.
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
    force_overwrite: Optional[bool] = False
