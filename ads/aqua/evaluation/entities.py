#!/usr/bin/env python
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
aqua.evaluation.entities
~~~~~~~~~~~~~~

This module contains dataclasses for aqua evaluation.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from ads.aqua.config.utils.serializer import Serializable
from ads.aqua.data import AquaResourceIdentifier


class CreateAquaEvaluationDetails(Serializable):
    """Class for creating aqua model evaluation.

    Properties
    ----------
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
    freeform_tags: (dict, optional)
        Freeform tags for the evaluation model
    defined_tags: (dict, optional)
        Defined tags for the evaluation model
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
    metrics: Optional[List[Dict[str, Any]]] = None
    force_overwrite: Optional[bool] = False
    freeform_tags: Optional[dict] = None
    defined_tags: Optional[dict] = None

    class Config:
        extra = "ignore"
        protected_namespaces = ()


class AquaEvalReport(Serializable):
    evaluation_id: str = ""
    content: str = ""

    class Config:
        extra = "ignore"


class AquaEvalParams(Serializable):
    shape: str = ""
    dataset_path: str = ""
    report_path: str = ""

    class Config:
        extra = "allow"


class AquaEvalMetric(Serializable):
    key: str
    name: str
    description: str = ""

    class Config:
        extra = "ignore"


class AquaEvalMetricSummary(Serializable):
    metric: str = ""
    score: str = ""
    grade: str = ""

    class Config:
        extra = "ignore"


class AquaEvalMetrics(Serializable):
    id: str
    report: str
    metric_results: List[AquaEvalMetric] = Field(default_factory=list)
    metric_summary_result: List[AquaEvalMetricSummary] = Field(default_factory=list)

    class Config:
        extra = "ignore"


class AquaEvaluationCommands(Serializable):
    evaluation_id: str
    evaluation_target_id: str
    input_data: Dict[str, Any]
    metrics: List[Dict[str, Any]]
    output_dir: str
    params: Dict[str, Any]

    class Config:
        extra = "ignore"


class AquaEvaluationSummary(Serializable):
    """Represents a summary of Aqua evalution."""

    id: str
    name: str
    console_url: str
    lifecycle_state: str
    lifecycle_details: str
    time_created: str
    tags: Dict[str, Any]
    experiment: AquaResourceIdentifier = Field(default_factory=AquaResourceIdentifier)
    source: AquaResourceIdentifier = Field(default_factory=AquaResourceIdentifier)
    job: AquaResourceIdentifier = Field(default_factory=AquaResourceIdentifier)
    parameters: AquaEvalParams = Field(default_factory=AquaEvalParams)

    class Config:
        extra = "ignore"


class AquaEvaluationDetail(AquaEvaluationSummary):
    """Represents a details of Aqua evalution."""

    log_group: AquaResourceIdentifier = Field(default_factory=AquaResourceIdentifier)
    log: AquaResourceIdentifier = Field(default_factory=AquaResourceIdentifier)
    introspection: dict = Field(default_factory=dict)
