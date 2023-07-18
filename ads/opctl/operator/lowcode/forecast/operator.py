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
class ForecastOperatorSpec(DataClassSerializable):
    name: str = None
    historical_data: InputData = field(default_factory=InputData)
    additional_data: InputData = field(default_factory=InputData)
    test_data: TestData = None
    output_directory: OutputDirectory = field(default_factory=OutputDirectory)
    report_file_name: str = None
    report_title: str = None
    report_theme: str = None
    report_metrics_name: str = None
    target_columns: List[str] = None
    datetime_column: DateTimeColumn = field(default_factory=DateTimeColumn)
    target_category_column: str = None
    horizon: Horizon = None
    model: str = None
    model_kwargs: Dict = None
    confidence_interval_width: float = None
    metric: str = None


@dataclass(repr=True)
class ForecastOperator(Operator):
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


def operate(operator_spec: ForecastOperator) -> None:
    """Runs the operator."""
    print("#" * 100)
    print(operator_spec.to_yaml())


def verify(spec: Dict) -> bool:
    """Verifies operator specification."""
    operator = ForecastOperator.from_dict(spec)
    print("*" * 100)
    print("The operator config has been successfully verified.")
    print(operator.to_yaml())
    print("*" * 100)
