#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from dataclasses import dataclass, field
from typing import Dict

from ads.common.serializer import DataClassSerializable
from ads.opctl.operator.common.operator_config import InputData, OperatorConfig, OutputDirectory
from ads.opctl.operator.common.utils import _load_yaml_from_uri
from ads.opctl.operator.lowcode.common.utils import find_output_dirname
from .const import SupportedMetrics, SupportedModels


@dataclass
class AutoScaling(DataClassSerializable):
    """Class representing simple autoscaling policy."""

    minimum_instance: int = 1
    maximum_instance: int = None
    cool_down_in_seconds: int = 600
    scale_in_threshold: int = 10
    scale_out_threshold: int = 80
    scaling_metric: str = "CPU_UTILIZATION"


@dataclass(repr=True)
class ModelDeploymentServer(DataClassSerializable):
    """Class representing model deployment server specification."""

    id: str = None
    display_name: str = None
    initial_shape: str = None
    description: str = None
    log_group: str = None
    log_id: str = None
    auto_scaling: AutoScaling = field(default_factory=AutoScaling)


@dataclass(repr=True)
class RegressionDeploymentConfig(DataClassSerializable):
    """Class representing regression model deployment settings."""

    model_catalog_display_name: str = None
    compartment_id: str = None
    project_id: str = None
    model_deployment: ModelDeploymentServer = field(
        default_factory=ModelDeploymentServer
    )


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
class RegressionOperatorSpec(DataClassSerializable):
    """Class representing regression operator specification."""

    training_data: InputData = field(default_factory=InputData)
    test_data: TestData = field(default_factory=TestData)
    output_directory: OutputDirectory = field(default_factory=OutputDirectory)

    target_column: str = None
    column_types: Dict = field(default_factory=dict)

    model: str = None
    model_kwargs: Dict = field(default_factory=dict)

    preprocessing: DataPreprocessor = field(default_factory=DataPreprocessor)
    metric: str = None

    training_predictions_filename: str = None
    test_predictions_filename: str = None
    training_metrics_filename: str = None
    test_metrics_filename: str = None
    report_filename: str = None
    report_title: str = None
    report_theme: str = None
    global_explanation_filename: str = None

    generate_report: bool = None
    generate_explanations: bool = None
    save_and_deploy_to_md: RegressionDeploymentConfig = None

    def __post_init__(self):
        """Adjusts defaults for the specification."""
        self.output_directory = self.output_directory or OutputDirectory(
            url=find_output_dirname(self.output_directory)
        )

        self.target_column = self.target_column or "target"
        self.metric = (self.metric or SupportedMetrics.RMSE).lower()

        self.model = self.model or SupportedModels.LINEAR_REGRESSION
        self.model_kwargs = self.model_kwargs or {}

        self.preprocessing = (
            self.preprocessing
            if self.preprocessing is not None
            else DataPreprocessor(enabled=True)
        )

        self.training_predictions_filename = (
            self.training_predictions_filename or "training_predictions.csv"
        )
        self.test_predictions_filename = (
            self.test_predictions_filename or "test_predictions.csv"
        )
        self.training_metrics_filename = (
            self.training_metrics_filename or "training_metrics.csv"
        )
        self.test_metrics_filename = self.test_metrics_filename or "test_metrics.csv"
        self.report_filename = self.report_filename or "report.html"
        self.report_title = self.report_title or "Regression Report"
        self.report_theme = self.report_theme or "light"
        self.global_explanation_filename = (
            self.global_explanation_filename or "global_explanations.csv"
        )

        self.generate_report = self.generate_report if self.generate_report is not None else True
        self.generate_explanations = (
            self.generate_explanations if self.generate_explanations is not None else False
        )
        if isinstance(self.save_and_deploy_to_md, bool):
            self.save_and_deploy_to_md = (
                RegressionDeploymentConfig() if self.save_and_deploy_to_md else None
            )
        elif self.save_and_deploy_to_md is not None and not isinstance(
            self.save_and_deploy_to_md, RegressionDeploymentConfig
        ):
            self.save_and_deploy_to_md = RegressionDeploymentConfig.from_dict(
                self.save_and_deploy_to_md
            )


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
