#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
aqua.evaluation.entities
~~~~~~~~~~~~~~

This module contains dataclasses for aqua evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

from ads.aqua.data import AquaResourceIdentifier
from ads.common.serializer import DataClassSerializable


@dataclass(repr=False)
class CreateAquaEvaluationDetails(DataClassSerializable):
    """Dataclass to create aqua model evaluation.

    Fields
    ------
    evaluation_source_id: str
        The evaluation source id. Must be either model or model deployment ocid.
    evaluation_name: str
        The name for evaluation.
    dataset_path: str
        The dataset path for the evaluation. Could be either a local path from notebook session
        or an object storage path.
    report_path: str
        The report path for the evaluation. Must be an object storage path.
    model_parameters: dict
        The parameters for the evaluation.
    shape_name: str
        The shape name for the evaluation job infrastructure.
    memory_in_gbs: float
        The memory in gbs for the shape selected.
    ocpus: float
        The ocpu count for the shape selected.
    block_storage_size: int
        The storage for the evaluation job infrastructure.
    compartment_id: (str, optional). Defaults to `None`.
        The compartment id for the evaluation.
    project_id: (str, optional). Defaults to `None`.
        The project id for the evaluation.
    evaluation_description: (str, optional). Defaults to `None`.
        The description for evaluation
    experiment_id: (str, optional). Defaults to `None`.
        The evaluation model version set id. If provided,
        evaluation model will be associated with it.
    experiment_name: (str, optional). Defaults to `None`.
        The evaluation model version set name. If provided,
        the model version set with the same name will be used if exists,
        otherwise a new model version set will be created with the name.
    experiment_description: (str, optional). Defaults to `None`.
        The description for the evaluation model version set.
    log_group_id: (str, optional). Defaults to `None`.
        The log group id for the evaluation job infrastructure.
    log_id: (str, optional). Defaults to `None`.
        The log id for the evaluation job infrastructure.
    metrics: (list, optional). Defaults to `None`.
        The metrics for the evaluation.
    force_overwrite: (bool, optional). Defaults to `False`.
        Whether to force overwrite the existing file in object storage.
    """

    evaluation_source_id: str
    evaluation_name: str
    dataset_path: str
    report_path: str
    model_parameters: dict
    shape_name: str
    block_storage_size: int
    compartment_id: Optional[str] = None
    project_id: Optional[str] = None
    evaluation_description: Optional[str] = None
    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None
    experiment_description: Optional[str] = None
    memory_in_gbs: Optional[float] = None
    ocpus: Optional[float] = None
    log_group_id: Optional[str] = None
    log_id: Optional[str] = None
    metrics: Optional[List] = None
    force_overwrite: Optional[bool] = False


@dataclass(repr=False)
class AquaEvalReport(DataClassSerializable):
    evaluation_id: str = ""
    content: str = ""


@dataclass(repr=False)
class ModelParams(DataClassSerializable):
    max_tokens: str = ""
    top_p: str = ""
    top_k: str = ""
    temperature: str = ""
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = field(default_factory=list)


@dataclass(repr=False)
class AquaEvalParams(ModelParams, DataClassSerializable):
    shape: str = ""
    dataset_path: str = ""
    report_path: str = ""


@dataclass(repr=False)
class AquaEvalMetric(DataClassSerializable):
    key: str
    name: str
    description: str = ""


@dataclass(repr=False)
class AquaEvalMetricSummary(DataClassSerializable):
    metric: str = ""
    score: str = ""
    grade: str = ""


@dataclass(repr=False)
class AquaEvalMetrics(DataClassSerializable):
    id: str
    report: str
    metric_results: List[AquaEvalMetric] = field(default_factory=list)
    metric_summary_result: List[AquaEvalMetricSummary] = field(default_factory=list)


@dataclass(repr=False)
class AquaEvaluationCommands(DataClassSerializable):
    evaluation_id: str
    evaluation_target_id: str
    input_data: dict
    metrics: list
    output_dir: str
    params: dict


@dataclass(repr=False)
class AquaEvaluationSummary(DataClassSerializable):
    """Represents a summary of Aqua evalution."""

    id: str
    name: str
    console_url: str
    lifecycle_state: str
    lifecycle_details: str
    time_created: str
    tags: dict
    experiment: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    source: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    job: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    parameters: AquaEvalParams = field(default_factory=AquaEvalParams)


@dataclass(repr=False)
class AquaEvaluationDetail(AquaEvaluationSummary, DataClassSerializable):
    """Represents a details of Aqua evalution."""

    log_group: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    log: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    introspection: dict = field(default_factory=dict)
