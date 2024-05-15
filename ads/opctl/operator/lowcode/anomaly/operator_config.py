#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from dataclasses import dataclass, field
from typing import Dict, List

from ads.common.serializer import DataClassSerializable
from ads.opctl.operator.common.utils import _load_yaml_from_uri
from ads.opctl.operator.common.operator_config import (
    OperatorConfig,
    OutputDirectory,
    InputData,
)
from .const import SupportedModels
from ads.opctl.operator.lowcode.common.utils import find_output_dirname


@dataclass(repr=True)
class ValidationData(InputData):
    """Class representing operator specification input data details."""


@dataclass(repr=True)
class DateTimeColumn(DataClassSerializable):
    """Class representing operator specification date time column details."""

    name: str = None
    format: str = None


@dataclass(repr=True)
class TestData(InputData):
    """Class representing operator specification test data details."""


@dataclass(repr=True)
class PreprocessingSteps(DataClassSerializable):
    """Class representing preprocessing steps for operator."""

    missing_value_imputation: bool = True
    outlier_treatment: bool = False


@dataclass(repr=True)
class DataPreprocessor(DataClassSerializable):
    """Class representing operator specification preprocessing details."""

    enabled: bool = True
    steps: PreprocessingSteps = field(default_factory=PreprocessingSteps)

@dataclass(repr=True)
class AnomalyOperatorSpec(DataClassSerializable):
    """Class representing operator specification."""

    input_data: InputData = field(default_factory=InputData)
    datetime_column: DateTimeColumn = field(default_factory=DateTimeColumn)
    test_data: TestData = field(default_factory=TestData)
    validation_data: ValidationData = field(default_factory=ValidationData)
    output_directory: OutputDirectory = field(default_factory=OutputDirectory)
    report_file_name: str = None
    report_title: str = None
    report_theme: str = None
    metrics_filename: str = None
    test_metrics_filename: str = None
    inliers_filename: str = None
    outliers_filename: str = None
    target_column: str = None
    target_category_columns: List[str] = field(default_factory=list)
    preprocessing: bool = None
    generate_report: bool = None
    generate_metrics: bool = None
    generate_inliers: bool = None
    model: str = None
    model_kwargs: Dict = field(default_factory=dict)
    contamination: float = None

    def __post_init__(self):
        """Adjusts the specification details."""
        self.output_directory = self.output_directory or OutputDirectory(url=find_output_dirname(self.output_directory))
        self.report_file_name = self.report_file_name or "report.html"
        self.report_theme = self.report_theme or "light"
        self.inliers_filename = self.inliers_filename or "inliers.csv"
        self.outliers_filename = self.outliers_filename or "outliers.csv"
        self.test_metrics_filename = self.test_metrics_filename or "metrics.csv"
        self.model = self.model or SupportedModels.Auto
        self.generate_inliers = (
            self.generate_inliers if self.generate_inliers is not None else False
        )
        self.model_kwargs = self.model_kwargs or dict()
        self.preprocessing = (
            self.preprocessing if self.preprocessing is not None else DataPreprocessor(enabled=True)
        )

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
