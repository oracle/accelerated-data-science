#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from copy import deepcopy
from typing import List, Dict

from ads.common import utils
from ads.feature_store.common.enums import ValidationEngineType, LevelType
from ads.feature_store.data_validation.great_expectation import ExpectationType
from ads.jobs.builders.base import Builder

logger = logging.getLogger(__name__)


class Expectation(Builder):
    _PREFIX = "Expectation_resource"

    CONST_NAME = "name"
    CONST_DESCRIPTION = "description"
    CONST_RULE_DETAILS = "createRuleDetails"
    CONST_EXPECTATION_TYPE = "expectationType"
    CONST_VALIDATION_ENGINE_TYPE = "validationEngineType"

    def __init__(
        self,
        name: str,
        rule_details: List["Rule"] = None,
        description: str = None,
        expectation_type=ExpectationType.LENIENT,
        validation_engine_type=ValidationEngineType.GREAT_EXPECTATIONS,
    ) -> None:
        """Initialize an expectation suite.

        Parameters
        ----------
        name : str, required
            The name of the expectation.
        rule_details : str, required
            The rules applies to feature definition, by default empty array.
        description : str, optional
            The description for expectation, by default None.
        expectation_type : ExpectationType, optional
            The expectation_type for expectation, by default LENIENT.
        validation_engine_type : ValidationEngineType, optional
            The validation_engine_type for expectation, by default GREAT_EXPECTATIONS.

        Methods
        -------
        with_rule_details(self, rule_details) -> Expectation
            Sets the rule details for expectation.
        with_expectation_type(self, expectation_type: ExpectationType) -> Expectation
            Sets the expectation_type for expectation.
        with_description(self, description: str) -> Expectation
            Sets the description for expectation.
        with_validation_engine_type(self, validation_engine_type: ValidationEngineType) -> Expectation
            Sets the validation_engine_type for expectation.
        """
        super().__init__()
        if not name:
            raise ValueError("Evaluation name must be specified.")

        self.set_spec(self.CONST_NAME, name)

        if rule_details:
            self.with_rule_details(rule_details)

        if description:
            self.with_description(description)
        self.with_expectation_type(expectation_type)
        self.with_validation_engine_type(validation_engine_type)

    @property
    def kind(self) -> str:
        """The kind of the object as showing in a YAML."""
        return "Expectation"

    @property
    def name(self) -> str:
        return self.get_spec(self.CONST_NAME)

    @property
    def description(self) -> str:
        return self.get_spec(self.CONST_DESCRIPTION)

    @description.setter
    def description(self, value: str):
        self.with_description(value)

    def with_description(self, description: str) -> "Expectation":
        """Sets the description.

        Parameters
        ----------
        description: str
            The description of the expectation.

        Returns
        -------
        Expectation
            The Expectation instance (self)
        """
        return self.set_spec(self.CONST_DESCRIPTION, description)

    @property
    def rule_details(self) -> str:
        return self.get_spec(self.CONST_RULE_DETAILS)

    @rule_details.setter
    def rule_details(self, value: str):
        self.with_description(value)

    def with_rule_details(self, rule_details: List["Rule"]) -> "Expectation":
        """Sets the rules for the expectation.

        Parameters
        ----------
        rule_details: str
            The rule_details of the expectation.

        Returns
        -------
        Expectation
            The Expectation instance (self)
        """
        return self.set_spec(
            self.CONST_RULE_DETAILS,
            [rule_detail.to_dict().get("spec") for rule_detail in rule_details],
        )

    @property
    def expectation_type(self) -> str:
        return self.get_spec(self.CONST_EXPECTATION_TYPE)

    @expectation_type.setter
    def expectation_type(self, expectation_type: ExpectationType):
        self.with_expectation_type(expectation_type)

    def with_expectation_type(self, expectation_type: ExpectationType) -> "Expectation":
        """Sets the dataset expectation_type .

        Parameters
        ----------
        expectation_type: str
            The expectation_type of the feature group.

        Returns
        -------
        Expectation
            The Expectation instance (self)
        """
        return self.set_spec(self.CONST_EXPECTATION_TYPE, expectation_type.value)

    @property
    def validation_engine_type(self) -> str:
        return self.get_spec(self.CONST_VALIDATION_ENGINE_TYPE)

    @validation_engine_type.setter
    def validation_engine_type(self, validation_engine_type: ValidationEngineType):
        self.with_validation_engine_type(validation_engine_type)

    def with_validation_engine_type(
        self, validation_engine_type: ValidationEngineType
    ) -> "Expectation":
        """Sets the validation_engine_type for the expectation.

        Parameters
        ----------
        validation_engine_type: str
            The validation_engine_type of the feature group.

        Returns
        -------
        Expectation
            The Expectation instance (self)
        """
        return self.set_spec(
            self.CONST_VALIDATION_ENGINE_TYPE, validation_engine_type.value
        )

    def to_dict(self) -> Dict:
        """Serializes rule to a dictionary.

        Returns
        -------
        dict
            The expectation resource serialized as a dictionary.
        """

        expectation_details = deepcopy(self._spec)
        return expectation_details

    def __repr__(self) -> str:
        """Displays the object as YAML."""
        return self.to_yaml()


class Rule(Builder):
    _PREFIX = "Rule_resource"

    CONST_NAME = "name"
    CONST_RULE_TYPE = "ruleType"
    CONST_LEVEL_TYPE = "levelType"
    CONST_ARGUMENTS = "arguments"

    def __init__(
        self,
        name: str,
        rule_type: str = None,
        arguments: dict = None,
        level_type: LevelType = LevelType.ERROR,
    ) -> None:
        """Initialize an expectation suite.

        Parameters
        ----------
        name : str, required
            The name of the expectation.
        rule_type : str, required
            The rule_type to expectation.
        arguments : Dict[str, str], optional
            The arguments for expectation, by default None
        level_type : str, optional
            The level_type for expectation, by default ERROR.

        Methods
        -------
        with_rule_type(self, rule_details) -> Rule
            Sets the rule type for rule.
        with_level_type(self, level_type: str) -> Rule
            Sets the level_type for rule.
        with_arguments(self, arguments: list[str]) -> Rule
            Sets the arguments for rule.
        """
        super().__init__()
        if not name:
            raise ValueError("Evaluation name must be specified.")
        self.set_spec(self.CONST_NAME, name)

        if self.rule_type:
            self.with_rule_type(rule_type)
        if self.arguments:
            self.with_arguments(arguments)

        self.with_level_type(level_type)

    @property
    def name(self) -> str:
        return self.get_spec(self.CONST_NAME)

    @property
    def rule_type(self) -> str:
        return self.get_spec(self.CONST_RULE_TYPE)

    @rule_type.setter
    def rule_type(self, rule_type: str) -> "Rule":
        return self.with_rule_type(rule_type)

    def with_rule_type(self, rule_type: str) -> "Rule":
        """Sets the rule_type.

        Parameters
        ----------
        rule_type: str
            The rule_type of expectation.

        Returns
        -------
        Rule
            The Rule instance (self)
        """
        return self.set_spec(self.CONST_RULE_TYPE, rule_type)

    @property
    def level_type(self) -> str:
        return self.get_spec(self.CONST_LEVEL_TYPE)

    @level_type.setter
    def level_type(self, level_type: LevelType) -> "Rule":
        return self.with_level_type(level_type)

    def with_level_type(self, level_type: LevelType) -> "Rule":
        """Sets the level_type.

        Parameters
        ----------
        level_type: LevelType
            The level_type of expectation.

        Returns
        -------
        Rule
            The Rule instance (self)
        """
        return self.set_spec(self.CONST_LEVEL_TYPE, level_type.value)

    @property
    def arguments(self) -> str:
        return self.get_spec(self.CONST_ARGUMENTS)

    @arguments.setter
    def arguments(self, arguments: dict) -> "Rule":
        return self.with_arguments(arguments)

    def with_arguments(self, arguments: dict) -> "Rule":
        """Sets the arguments.

        Parameters
        ----------
        arguments: list[str]
            The arguments of expectation.

        Returns
        -------
        Rule
            The Rule instance (self)
        """
        return self.set_spec(self.CONST_ARGUMENTS, arguments)

    def to_dict(self) -> Dict:
        """Serializes rule to a dictionary.

        Returns
        -------
        dict
            The rule resource serialized as a dictionary.
        """

        spec = deepcopy(self._spec)
        for key, value in spec.items():
            if hasattr(value, "to_dict"):
                value = value.to_dict()
            spec[key] = value

        return {
            "kind": self.kind,
            "type": self.type,
            "spec": utils.batch_convert_case(spec, "camel"),
        }

    def __repr__(self) -> str:
        """Displays the object as YAML."""
        return self.to_yaml()
