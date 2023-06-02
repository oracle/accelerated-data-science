#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from copy import deepcopy
from typing import Dict, List

from ads.jobs.builders.base import Builder


class StatisticsConfig(Builder):
    """Sets the Statistics Config Details.
    Methods
    -------
        with_enabled(self, enabled: bool) -> "StatisticsConfig"
        Sets True/False for enabled
        with_columns(self, columns: List[str]) -> "StatisticsConfig"
        Sets the column names for the statistics config
    """

    CONST_ENABLED = "isEnabled"
    CONST_COLUMNS = "columns"

    attribute_map = {
        CONST_ENABLED: "is_enabled",
        CONST_COLUMNS: "columns",
    }

    def __init__(self, is_enabled: bool = True, columns: List[str] = None) -> None:
        super().__init__()

        if columns is None:
            columns = []
        self.with_is_enabled(is_enabled)
        if columns:
            self.with_columns(columns)

    @property
    def is_enabled(self) -> bool:
        return self.get_spec(self.CONST_ENABLED)

    @is_enabled.setter
    def is_enabled(self, is_enabled: bool):
        self.with_is_enabled(is_enabled)

    def with_is_enabled(self, is_enabled: bool) -> "StatisticsConfig":
        """Sets True/False for enabled

        Parameters
        ----------
        is_enabled: bool
            enable or disable the statistics computation

        Returns
        -------
        StatisticsConfig
            The StatisticsConfig instance (self)
        """
        return self.set_spec(self.CONST_ENABLED, is_enabled)

    @property
    def columns(self) -> List[str]:
        return self.get_spec(self.CONST_COLUMNS)

    @columns.setter
    def columns(self, columns: List[str]):
        self.with_columns(columns)

    def with_columns(self, columns: List[str]) -> "StatisticsConfig":
        """Sets the columns for which the stats to be calculated .

        Parameters
        ----------
        columns: List[str]
            columns for which the stats to be calculated.

        Returns
        -------
        StatisticsConfig
            The StatisticsConfig instance (self)
        """
        return self.set_spec(self.CONST_COLUMNS, columns)

    @property
    def kind(self) -> str:
        """The kind of the object as showing in a YAML."""
        return "statistics_config"

    def to_dict(self) -> Dict:
        """Serializes rule to a dictionary.

        Returns
        -------
        dict
            The rule resource serialized as a dictionary.
        """

        spec = deepcopy(self._spec)
        return spec
