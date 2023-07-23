#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from dataclasses import dataclass, field
from typing import Dict, List

from ads.common.serializer import DataClassSerializable
from ads.opctl.operator.common.utils import _load_yaml_from_uri
from ads.opctl.operator.operator import Operator

from .const import SupportedMetrics
from .model.factory import ForecastOperatorModelFactory


@dataclass(repr=True)
class InputData(DataClassSerializable):
    format: str = None
    columns: List[str] = None
    url: str = None
    options: Dict = None
    limit: int = None


@dataclass(repr=True)
class TestData(DataClassSerializable):
    connect_args: Dict = None
    format: str = None
    url: str = None
    name: str = None
    options: Dict = None


@dataclass(repr=True)
class OutputDirectory(DataClassSerializable):
    connect_args: Dict = None
    format: str = None
    url: str = None
    name: str = None
    options: Dict = None


@dataclass(repr=True)
class DateTimeColumn(DataClassSerializable):
    name: str = None
    format: str = None


@dataclass(repr=True)
class Horizon(DataClassSerializable):
    periods: int = None
    interval: int = None
    interval_unit: str = None


@dataclass(repr=True)
class Tuning(DataClassSerializable):
    n_trials: int = None


@dataclass(repr=True)
class ForecastOperatorSpec(DataClassSerializable):
    """The dataclass representing forecasting operator specification."""

    name: str = None
    historical_data: InputData = field(default_factory=InputData)
    additional_data: InputData = field(default_factory=InputData)
    test_data: TestData = field(default_factory=TestData)
    output_directory: OutputDirectory = field(default_factory=OutputDirectory)
    report_file_name: str = None
    report_title: str = None
    report_theme: str = None
    report_metrics_name: str = None
    target_column: str = None
    datetime_column: DateTimeColumn = field(default_factory=DateTimeColumn)
    target_category_columns: List[str] = field(default_factory=list)
    horizon: Horizon = field(default_factory=Horizon)
    model: str = None
    model_kwargs: Dict = field(default_factory=dict)
    confidence_interval_width: float = None
    metric: str = None
    tuning: Tuning = field(default_factory=Tuning)

    def __post_init__(self):
        # setup default values
        self.metric = (self.metric or "").lower() or SupportedMetrics.SMAPE
        # self.confidence_interval_width = self.confidence_interval_width or 0.80
        self.report_file_name = self.report_file_name or "report.html"
        self.report_theme = self.report_theme or "light"
        self.report_metrics_name = self.report_metrics_name or "report.csv"
        self.target_column = self.target_column or "Sales"


@dataclass(repr=True)
class ForecastOperatorConfig(Operator):
    kind: str = "operator"
    type: str = "forecast"
    version: str = "v1"
    spec: ForecastOperatorSpec = field(default_factory=ForecastOperatorSpec)

    @classmethod
    def _load_schema(cls) -> str:
        """Loads operator's schema."""
        return _load_yaml_from_uri(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.yaml")
        )


def operate(operator_config: ForecastOperatorConfig) -> None:
    """Runs the forecasting operator."""
    ForecastOperatorModelFactory.get_model(operator_config).generate_report()


def verify(spec: Dict) -> bool:
    """Verifies operator specification."""
    operator = ForecastOperatorConfig.from_dict(spec)
    msg_header = (
        f"{'*' * 50} The operator config has been successfully verified {'*' * 50}"
    )
    print(operator.to_yaml())
    print("*" * len(msg_header))
