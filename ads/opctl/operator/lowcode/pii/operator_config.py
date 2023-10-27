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
class OutputDirectory(DataClassSerializable):
    """Class representing operator specification output directory details."""

    connect_args: Dict = None
    format: str = None
    url: str = None
    name: str = None
    options: Dict = None


@dataclass(repr=True)
class PIIOperatorSpec(DataClassSerializable):
    """Class representing PII operator specification."""

    name: str = None
    input_data: InputData = field(default_factory=InputData)
    output_directory: OutputDirectory = field(default_factory=OutputDirectory)
    report_file_name: str = None
    report_title: str = None
    report_theme: str = None

    def __post_init__(self):
        """Adjusts the specification details."""
        self.report_file_name = self.report_file_name or "report.html"
        self.report_theme = self.report_theme or "light"


@dataclass(repr=True)
class PIIOperatorConfig(OperatorConfig):
    """Class representing PII operator config.

    Attributes
    ----------
    kind: str
        The kind of the resource. For operators it is always - `operator`.
    type: str
        The type of the operator. For PII operator it is always - `PII`
    version: str
        The version of the operator.
    spec: PIIOperatorSpec
        The PII operator specification.
    """

    kind: str = "operator"
    type: str = "PII"
    version: str = "v1"
    spec: PIIOperatorSpec = field(default_factory=PIIOperatorSpec)

    @classmethod
    def _load_schema(cls) -> str:
        """Loads operator schema."""
        return _load_yaml_from_uri(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.yaml")
        )
