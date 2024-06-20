#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from dataclasses import dataclass, field

from ads.common.serializer import DataClassSerializable
from ads.opctl.operator.common.operator_config import OperatorConfig, InputData
from ads.opctl.operator.common.utils import _load_yaml_from_uri
from ads.opctl.operator.lowcode.common.utils import find_output_dirname
from .constant import SupportedModels


@dataclass(repr=True)
class OutputDirectory(DataClassSerializable):
    """Class representing operator specification output directory details."""

    url: str = None
    name: str = None


@dataclass(repr=True)
class RecommenderOperatorSpec(DataClassSerializable):
    """Class representing Recommender operator specification."""

    user_data: InputData = field(default_factory=InputData)
    item_data: InputData = field(default_factory=InputData)
    interactions_data: InputData = field(default_factory=InputData)
    output_directory: OutputDirectory = field(default_factory=OutputDirectory)
    top_k: int = None
    model_name: str = None
    user_column: str = None
    item_column: str = None
    interaction_column: str = None
    recommendations_filename: str = None
    generate_report: bool = None
    report_filename: str = None


    def __post_init__(self):
        """Adjusts the specification details."""
        self.output_directory = self.output_directory or OutputDirectory(url=find_output_dirname(self.output_directory))
        self.model_name = self.model_name or SupportedModels.SVD
        self.recommendations_filename = self.recommendations_filename or "recommendations.csv"
        # For Report Generation. When user doesn't specify defaults to True
        self.generate_report = (
            self.generate_report if self.generate_report is not None else True
        )
        self.report_filename = self.report_filename or "report.html"


@dataclass(repr=True)
class RecommenderOperatorConfig(OperatorConfig):
    """Class representing Recommender operator config.

    Attributes
    ----------
    kind: str
        The kind of the resource. For operators it is always - `operator`.
    type: str
        The type of the operator. For Recommender operator it is always - `Recommender`
    version: str
        The version of the operator.
    spec: RecommenderOperatorSpec
        The Recommender operator specification.
    """

    kind: str = "operator"
    type: str = "Recommender"
    version: str = "v1"
    spec: RecommenderOperatorSpec = field(default_factory=RecommenderOperatorSpec)

    @classmethod
    def _load_schema(cls) -> str:
        """Loads operator schema."""
        return _load_yaml_from_uri(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.yaml")
        )
