#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
aqua.evaluation.response_model
~~~~~~~~~~~~~~

This module contains response models for aqua evaluation apis.
"""

from dataclasses import InitVar, dataclass, field
from typing import List, Optional, Union

from ads.aqua.data import AquaResourceIdentifier
from ads.common.serializer import DataClassSerializable
from ads.common.utils import get_console_link


@dataclass(repr=False)
class AquaEvaluationCommands(DataClassSerializable):
    evaluation_id: str
    evaluation_target_id: str
    input_data: dict
    metrics: list
    output_dir: str
    params: dict


@dataclass(repr=False)
class AquaEvalReport(DataClassSerializable):
    """Represents GET evaluation/<id>/report response model."""

    evaluation_id: str = ""
    content: str = ""


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
    """Represents GET evaluation/<id>/metrics response model."""

    id: str
    report: str
    metric_results: List[AquaEvalMetric] = field(default_factory=list)
    metric_summary_result: List[AquaEvalMetricSummary] = field(default_factory=list)


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

    def set_shape(self, value):
        self.shape = value


@dataclass(repr=False)
class AquaEvaluationSummary(DataClassSerializable):
    """Represents GET "evaluations" response model."""

    id: str = None
    name: str = None
    console_url: str = None
    lifecycle_state: str = None
    lifecycle_details: str = None
    time_created: str = None
    tags: dict = None
    experiment: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    source: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    job: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    parameters: AquaEvalParams = field(default_factory=AquaEvalParams)

    region: InitVar = None

    def __post_init__(self, region):
        if self.id and not self.console_url:
            self.console_url = get_console_link(
                resource="models",
                ocid=self.id,
                region=region,
            )


@dataclass(repr=False)
class AquaEvaluationDetail(AquaEvaluationSummary, DataClassSerializable):
    """Represents GET "evaluation/<id>" response model."""

    log_group: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    log: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    introspection: dict = field(default_factory=dict)


@dataclass
class AquaEvaluationStatus(DataClassSerializable):
    """Represents GET evaluation/<id>/status response model."""

    id: str = ""
    lifecycle_state: str = ""
    lifecycle_details: str = ""
    log_id: str = ""
    log_url: str = ""
    loggroup_id: str = ""
    loggroup_url: str = ""
