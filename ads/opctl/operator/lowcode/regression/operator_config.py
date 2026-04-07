#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import ast
from dataclasses import dataclass, field
from typing import Dict, List

from ads.common.serializer import DataClassSerializable
from ads.opctl.operator.common.operator_config import InputData, OperatorConfig, OutputDirectory
from ads.opctl.operator.common.utils import _load_yaml_from_uri
from ads.opctl.operator.lowcode.common.utils import find_output_dirname

from .const import SupportedMetrics, SupportedModels


@dataclass(repr=True)
class ValidationData(InputData):
    """Class representing optional validation data details."""


@dataclass(repr=True)
class TestData(InputData):
    """Class representing optional test data details."""


@dataclass(repr=True)
class PreprocessingSteps(DataClassSerializable):
    """Class representing preprocessing steps for operator."""

    missing_value_imputation: bool = True
    categorical_encoding: bool = True


@dataclass(repr=True)
class DataPreprocessor(DataClassSerializable):
    """Class representing preprocessing config for operator."""

    enabled: bool = True
    steps: PreprocessingSteps = field(default_factory=PreprocessingSteps)


@dataclass(repr=True)
class ModelSpec(DataClassSerializable):
    """Class representing model selection and hyperparameters."""

    name: str = None
    params: Dict = field(default_factory=dict)


@dataclass(repr=True)
class RegressionOperatorSpec(DataClassSerializable):
    """Class representing regression operator specification."""

    training_data: InputData = field(default_factory=InputData)
    validation_data: ValidationData = field(default_factory=ValidationData)
    test_data: TestData = field(default_factory=TestData)
    output_directory: OutputDirectory = field(default_factory=OutputDirectory)

    target_column: str = None
    feature_columns: List[str] = field(default_factory=list)
    column_types: Dict = field(default_factory=dict)

    model: ModelSpec = field(default_factory=ModelSpec)
    model_kwargs: Dict = field(default_factory=dict)

    preprocessing: DataPreprocessor = field(default_factory=DataPreprocessor)
    metric: str = None

    predictions_filename: str = None
    metrics_filename: str = None
    test_metrics_filename: str = None
    report_filename: str = None
    report_title: str = None
    report_theme: str = None
    feature_importance_filename: str = None
    global_explanation_filename: str = None
    local_explanation_filename: str = None

    generate_report: bool = None
    generate_explanations: bool = None
    generate_model_pickle: bool = None
    deploy_to_md: bool = None

    def __post_init__(self):
        """Adjusts defaults for the specification."""
        self.output_directory = self.output_directory or OutputDirectory(
            url=find_output_dirname(self.output_directory)
        )

        self.target_column = self.target_column or "target"
        self.metric = (self.metric or SupportedMetrics.RMSE).lower()

        self.model = self.model or ModelSpec()
        if isinstance(self.model, str):
            parsed = None
            try:
                parsed = ast.literal_eval(self.model)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                self.model = ModelSpec(
                    name=parsed.get("name"),
                    params=parsed.get("params") or {},
                )
            else:
                self.model = ModelSpec(name=self.model)

        model_name = self.model.name or SupportedModels.RANDOM_FOREST
        if isinstance(model_name, str):
            model_name = model_name.lower()
        self.model.name = model_name
        self.model.params = self.model.params or {}
        self.model_kwargs = self.model_kwargs or {}

        self.preprocessing = (
            self.preprocessing
            if self.preprocessing is not None
            else DataPreprocessor(enabled=True)
        )

        self.predictions_filename = self.predictions_filename or "predictions.csv"
        self.metrics_filename = self.metrics_filename or "metrics.csv"
        self.test_metrics_filename = self.test_metrics_filename or "test_metrics.csv"
        self.report_filename = self.report_filename or "report.html"
        self.report_title = self.report_title or "Regression Report"
        self.report_theme = self.report_theme or "light"
        self.feature_importance_filename = (
            self.feature_importance_filename or "feature_importance.csv"
        )
        self.global_explanation_filename = (
            self.global_explanation_filename or "global_explanations.csv"
        )
        self.local_explanation_filename = (
            self.local_explanation_filename or "local_explanations.csv"
        )

        self.generate_report = self.generate_report if self.generate_report is not None else True
        self.generate_explanations = (
            self.generate_explanations if self.generate_explanations is not None else False
        )
        self.generate_model_pickle = (
            self.generate_model_pickle if self.generate_model_pickle is not None else False
        )
        self.deploy_to_md = self.deploy_to_md if self.deploy_to_md is not None else False


@dataclass(repr=True)
class RegressionOperatorConfig(OperatorConfig):
    """Class representing regression operator config."""

    kind: str = "operator"
    type: str = "regression"
    version: str = "v1"
    spec: RegressionOperatorSpec = field(default_factory=RegressionOperatorSpec)

    @classmethod
    def _load_schema(cls) -> str:
        """Loads operator schema."""
        return _load_yaml_from_uri(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.yaml")
        )
