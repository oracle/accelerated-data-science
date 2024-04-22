#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from dataclasses import dataclass, field
from typing import Dict, List

from ads.common.serializer import DataClassSerializable
from ads.opctl.operator.common.operator_config import OperatorConfig, InputData
from ads.opctl.operator.common.utils import _load_yaml_from_uri
from ads.opctl.operator.lowcode.pii.constant import (
    DEFAULT_SHOW_ROWS,
    DEFAULT_REPORT_FILENAME,
    DEFAULT_TARGET_COLUMN,
)


@dataclass(repr=True)
class OutputDirectory(DataClassSerializable):
    """Class representing operator specification output directory details."""

    url: str = None
    name: str = None


@dataclass(repr=True)
class Report(DataClassSerializable):
    """Class representing operator specification report details."""

    report_filename: str = None
    show_rows: int = None
    show_sensitive_content: bool = False


@dataclass(repr=True)
class Detector(DataClassSerializable):
    """Class representing operator specification redactor directory details."""

    name: str = None
    action: str = None


@dataclass(repr=True)
class PiiOperatorSpec(DataClassSerializable):
    """Class representing pii operator specification."""

    input_data: InputData = field(default_factory=InputData)
    output_directory: OutputDirectory = field(default_factory=OutputDirectory)
    report: Report = field(default_factory=Report)
    target_column: str = None
    detectors: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        """Adjusts the specification details."""

        self.target_column = self.target_column or DEFAULT_TARGET_COLUMN
        self.report = self.report or Report.from_dict(
            {
                "report_filename": DEFAULT_REPORT_FILENAME,
                "show_rows": DEFAULT_SHOW_ROWS,
                "show_sensitive_content": False,
            }
        )


@dataclass(repr=True)
class PiiOperatorConfig(OperatorConfig):
    """Class representing pii operator config.

    Attributes
    ----------
    kind: str
        The kind of the resource. For operators it is always - `operator`.
    type: str
        The type of the operator. For pii operator it is always - `pii`
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
