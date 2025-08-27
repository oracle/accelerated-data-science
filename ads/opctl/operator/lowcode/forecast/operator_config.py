#!/usr/bin/env python

# Copyright (c) 2023, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from dataclasses import dataclass, field
from typing import Dict, List

from ads.common.serializer import DataClassSerializable
from ads.opctl.operator.common.operator_config import (
    InputData,
    OperatorConfig,
    OutputDirectory,
)
from ads.opctl.operator.common.utils import _load_yaml_from_uri
from ads.opctl.operator.lowcode.common.utils import find_output_dirname

from .const import SpeedAccuracyMode, SupportedMetrics, SupportedModels


@dataclass
class AutoScaling(DataClassSerializable):
    """Class representing simple autoscaling policy"""

    minimum_instance: int = 1
    maximum_instance: int = None
    cool_down_in_seconds: int = 600
    scale_in_threshold: int = 10
    scale_out_threshold: int = 80
    scaling_metric: str = "CPU_UTILIZATION"


@dataclass(repr=True)
class ModelDeploymentServer(DataClassSerializable):
    """Class representing model deployment server specification for whatif-analysis."""

    display_name: str = None
    initial_shape: str = None
    description: str = None
    log_group: str = None
    log_id: str = None
    auto_scaling: AutoScaling = field(default_factory=AutoScaling)


@dataclass(repr=True)
class WhatIfAnalysis(DataClassSerializable):
    """Class representing operator specification for whatif-analysis."""

    model_display_name: str = None
    compartment_id: str = None
    project_id: str = None
    model_deployment: ModelDeploymentServer = field(
        default_factory=ModelDeploymentServer
    )


@dataclass(repr=True)
class TestData(InputData):
    """Class representing operator specification test data details."""


@dataclass(repr=True)
class DateTimeColumn(DataClassSerializable):
    """Class representing operator specification date time column details."""

    name: str = None
    format: str = None


@dataclass(repr=True)
class PreprocessingSteps(DataClassSerializable):
    """Class representing preprocessing steps for operator."""

    missing_value_imputation: bool = True
    outlier_treatment: bool = True


@dataclass(repr=True)
class DataPreprocessor(DataClassSerializable):
    """Class representing operator specification preprocessing details."""

    enabled: bool = True
    steps: PreprocessingSteps = field(default_factory=PreprocessingSteps)


@dataclass(repr=True)
class Tuning(DataClassSerializable):
    """Class representing operator specification tuning details."""

    n_trials: int = None


@dataclass(repr=True)
class ForecastOperatorSpec(DataClassSerializable):
    """
    Class representing forecast operator specification.

    Attributes
    ----------
    name: str
        The name of the forecast operator.
    historical_data: InputData
        The historical data to be used for forecasting.
    additional_data: InputData
        Additional data to be used for forecasting.
    test_data: TestData
        The test data to be used for evaluating the forecast.
    output_directory: OutputDirectory
        The directory where the output files will be saved.
    report_filename: str
        The name of the report file. Defaults to "report.html".
    report_title: str
        The title of the report.
    report_theme: str
        The theme of the report. Can be "light" or "dark". Defaults to "light".
    metrics_filename: str
        The name of the metrics file. Defaults to "metrics.csv".
    test_metrics_filename: str
        The name of the test metrics file. Defaults to "test_metrics.csv".
    forecast_filename: str
        The name of the forecast file. Defaults to "forecast.csv".
    global_explanation_filename: str
        The name of the global explanation file. Defaults to "global_explanation.csv".
    local_explanation_filename: str
        The name of the local explanation file. Defaults to "local_explanation.csv".
    target_column: str
        The name of the target column.
    preprocessing: DataPreprocessor
        The data preprocessing settings.
    datetime_column: DateTimeColumn
        The datetime column details.
    target_category_columns: List[str]
        The list of target category columns.
    generate_report: bool
        Whether to generate a report. Defaults to True.
    generate_forecast_file: bool
        Whether to generate a forecast file. Defaults to True.
    generate_metrics: bool
        Whether to generate metrics. Defaults to True.
    generate_metrics_file: bool
        Whether to generate a metrics file. Defaults to True.
    generate_explanations: bool
        Whether to generate explanations. Defaults to False.
    generate_explanation_files: bool
        Whether to generate explanation files. Defaults to True.
    explanations_accuracy_mode: str
        The accuracy mode for explanations. Can be "HIGH_ACCURACY", "BALANCED", "FAST_APPROXIMATE", or "AUTOMLX".
    horizon: int
        The forecast horizon.
    model: str
        The forecasting model to be used.
    model_kwargs: Dict
        The keyword arguments for the model.
    model_parameters: str
        The model parameters.
    previous_output_dir: str
        The directory of a previous run to be used for forecasting.
    generate_model_parameters: bool
        Whether to generate model parameters. Defaults to False.
    generate_model_pickle: bool
        Whether to generate a model pickle. Defaults to False.
g    confidence_interval_width: float
        The width of the confidence interval. Defaults to 0.80.
    metric: str
        The metric to be used for evaluation.
    tuning: Tuning
        The tuning settings.
    what_if_analysis: WhatIfAnalysis
        The what-if analysis settings.
    """

    name: str = None
    historical_data: InputData = field(default_factory=InputData)
    additional_data: InputData = field(default_factory=InputData)
    test_data: TestData = field(default_factory=TestData)
    output_directory: OutputDirectory = field(default_factory=OutputDirectory)
    report_filename: str = None
    report_title: str = None
    report_theme: str = None
    metrics_filename: str = None
    test_metrics_filename: str = None
    forecast_filename: str = None
    global_explanation_filename: str = None
    local_explanation_filename: str = None
    target_column: str = None
    preprocessing: DataPreprocessor = field(default_factory=DataPreprocessor)
    datetime_column: DateTimeColumn = field(default_factory=DateTimeColumn)
    target_category_columns: List[str] = field(default_factory=list)
    generate_report: bool = None
    generate_forecast_file: bool = None
    generate_metrics: bool = None
    generate_metrics_file: bool = None
    generate_explanations: bool = None
    generate_explanation_files: bool = None
    explanations_accuracy_mode: str = None
    horizon: int = None
    model: str = None
    model_kwargs: Dict = field(default_factory=dict)
    model_parameters: str = None
    previous_output_dir: str = None
    generate_model_parameters: bool = None
    generate_model_pickle: bool = None
    confidence_interval_width: float = None
    metric: str = None
    tuning: Tuning = field(default_factory=Tuning)
    what_if_analysis: WhatIfAnalysis = field(default_factory=WhatIfAnalysis)

    def __post_init__(self):
        """Adjusts the specification details."""
        self.output_directory = self.output_directory or OutputDirectory(
            url=find_output_dirname(self.output_directory)
        )
        self.generate_model_pickle = self.generate_model_pickle or self.what_if_analysis
        self.metric = (self.metric or "").lower() or SupportedMetrics.SMAPE.lower()
        self.model = self.model or SupportedModels.Prophet
        self.confidence_interval_width = self.confidence_interval_width or 0.80
        self.report_filename = self.report_filename or "report.html"
        self.preprocessing = (
            self.preprocessing
            if self.preprocessing is not None
            else DataPreprocessor(enabled=True)
        )
        # For Report Generation. When user doesn't specify defaults to True
        self.generate_report = (
            self.generate_report if self.generate_report is not None else True
        )
        # For Metrics files Generation. When user doesn't specify defaults to True
        self.generate_metrics = (
            self.generate_metrics if self.generate_metrics is not None else True
        )
        self.generate_metrics_file = (
            self.generate_metrics_file
            if self.generate_metrics_file is not None
            else True
        )
        self.generate_forecast_file = (
            self.generate_forecast_file
            if self.generate_forecast_file is not None
            else True
        )
        self.generate_explanation_files = (
            self.generate_explanation_files
            if self.generate_explanation_files is not None
            else True
        )
        # For Explanations Generation. When user doesn't specify defaults to False
        self.generate_explanations = (
            self.generate_explanations
            if self.generate_explanations is not None
            else False
        )
        self.explanations_accuracy_mode = (
            self.explanations_accuracy_mode or SpeedAccuracyMode.FAST_APPROXIMATE
        )

        self.generate_model_parameters = (
            self.generate_model_parameters
            if self.generate_model_parameters is not None
            else False
        )
        self.generate_model_pickle = (
            self.generate_model_pickle
            if self.generate_model_pickle is not None
            else False
        )
        self.report_title = self.report_title or "Forecast Report"
        self.report_theme = self.report_theme or "light"
        self.metrics_filename = self.metrics_filename or "metrics.csv"
        self.test_metrics_filename = self.test_metrics_filename or "test_metrics.csv"
        self.forecast_filename = self.forecast_filename or "forecast.csv"
        self.global_explanation_filename = (
            self.global_explanation_filename or "global_explanation.csv"
        )
        self.local_explanation_filename = (
            self.local_explanation_filename or "local_explanation.csv"
        )
        self.target_column = self.target_column or "Sales"
        self.errors_dict_filename = "errors.json"
        self.model_kwargs = self.model_kwargs or {}


@dataclass(repr=True)
class ForecastOperatorConfig(OperatorConfig):
    """Class representing forecast operator config.

    Attributes
    ----------
    kind: str
        The kind of the resource. For operators it is always - `operator`.
    type: str
        The type of the operator. For forecast operator it is always - `forecast`
    version: str
        The version of the operator.
    spec: ForecastOperatorSpec
        The forecast operator specification.
    """

    kind: str = "operator"
    type: str = "forecast"
    version: str = "v1"
    spec: ForecastOperatorSpec = field(default_factory=ForecastOperatorSpec)

    @classmethod
    def _load_schema(cls) -> str:
        """Loads operator schema."""
        return _load_yaml_from_uri(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.yaml")
        )
