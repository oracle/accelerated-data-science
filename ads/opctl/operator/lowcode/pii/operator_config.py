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
    url: str = None


@dataclass(repr=True)
class OutputDirectory(DataClassSerializable):
    """Class representing operator specification output directory details."""

    url: str = None
    name: str = None


@dataclass(repr=True)
class Report(DataClassSerializable):
    """Class representing operator specification report details."""

    report_filename: str = None
    show_rows: int = 25
    show_sensitive_content: bool = False


@dataclass(repr=True)
class Redactor(DataClassSerializable):
    """Class representing operator specification redactor directory details."""

    detectors: List[str] = None
    # TODO:
    spacy_detectors: Dict = None
    anonymization: List[str] = None


@dataclass(repr=True)
class PiiOperatorSpec(DataClassSerializable):
    """Class representing pii operator specification."""

    name: str = None
    input_data: InputData = field(default_factory=InputData)
    output_directory: OutputDirectory = field(default_factory=OutputDirectory)
    report: Report = field(default_factory=Report)
    target_column: str = None
    redactor: Redactor = field(default_factory=Redactor)

    def __post_init__(self):
        """Adjusts the specification details."""
        # self.report_file_name = self.report_file_name or "report.html"
        self.target_column = self.target_column or "target"
        self.report.report_filename = self.report.report_filename or "report.html"
        self.report.show_rows = self.report.show_rows or 25
        self.report.show_sensitive_content = self.report.show_sensitive_content or False
        self.output_directory.url = self.output_directory.url or "result/"
        self.output_directory.name = self.output_directory.name or os.path.basename(
            self.input_data.url
        )


@dataclass(repr=True)
class PiiOperatorConfig(OperatorConfig):
    """Class representing pii operator config.

    Attributes
    ----------
    kind: str
        The kind of the resource. For operators it is always - `operator`.
    type: str
        The type of the operator. For pii operator it is always - `forecast`
    version: str
        The version of the operator.
    spec: PiiOperatorSpec
        The pii operator specification.
    """

    kind: str = "operator"
    type: str = "pii"
    version: str = "v1"
    spec: PiiOperatorSpec = field(default_factory=PiiOperatorSpec)

    @classmethod
    def _load_schema(cls) -> str:
        """Loads operator schema."""
        return _load_yaml_from_uri(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.yaml")
        )
