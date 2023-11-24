#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from dataclasses import dataclass, field
from typing import Dict, List

from ads.common.serializer import DataClassSerializable
from ads.opctl.operator.common.utils import _load_yaml_from_uri
from ads.opctl.operator.common.operator_config import OperatorConfig


@dataclass(repr=True)
class InputData(DataClassSerializable):
    """Class representing operator specification input data details."""

    format: str = None
    columns: List[str] = None
    url: str = None
    options: Dict = None
    limit: int = None


@dataclass(repr=True)
class DateTimeColumn(DataClassSerializable):
    """Class representing operator specification date time column details."""

    name: str = None
    format: str = None


@dataclass(repr=True)
class TestData(DataClassSerializable):
    """Class representing operator specification test data details."""

    connect_args: Dict = None
    format: str = None
    columns: List[str] = None
    url: str = None
    name: str = None
    options: Dict = None


@dataclass(repr=True)
class OutputDirectory(DataClassSerializable):
    """Class representing operator specification output directory details."""

    connect_args: Dict = None
    format: str = None
    url: str = None
    name: str = None
    options: Dict = None


@dataclass(repr=True)
class AnomalyOperatorSpec(DataClassSerializable):
    """Class representing operator specification."""

    name: str = None
    historical_data: InputData = field(default_factory=InputData)
    datetime_column: DateTimeColumn = field(default_factory=DateTimeColumn)
    test_data: TestData = field(default_factory=TestData)
    output_directory: OutputDirectory = field(default_factory=OutputDirectory)
    report_file_name: str = None
    report_title: str = None
    report_theme: str = None
    metrics_filename: str = None
    test_metrics_filename: str = None
    forecast_filename: str = None
    global_explanation_filename: str = None
    local_explanation_filename: str = None
    target_column: str = None
    target_category_columns: List[str] = field(default_factory=list)
    preprocessing: bool = None
    generate_report: bool = None
    generate_metrics: bool = None
    generate_explanations: bool = None
    model: str = None
    model_kwargs: Dict = field(default_factory=dict)
    metric: str = None

    def __post_init__(self):
        """Adjusts the specification details."""
        self.report_file_name = self.report_file_name or "report.html"
        self.report_theme = self.report_theme or "light"
        self.model_kwargs = self.model_kwargs or dict()


@dataclass(repr=True)
class AnomalyOperatorConfig(OperatorConfig):
    """Class representing operator config.

    Attributes
    ----------
    kind: str
        The kind of the resource. For operators it is always - `operator`.
    type: str
        The type of the operator.
    version: str
        The version of the operator.
    spec: AnomalyOperatorSpec
        The operator specification.
    """

    kind: str = "operator"
    type: str = "anomaly"
    version: str = "v1"
    spec: AnomalyOperatorSpec = field(default_factory=AnomalyOperatorSpec)

    @classmethod
    def _load_schema(cls) -> str:
        """Loads operator schema."""
        return _load_yaml_from_uri(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.yaml")
        )
