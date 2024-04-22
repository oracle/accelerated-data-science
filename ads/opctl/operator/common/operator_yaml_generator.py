#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl.operator.common.utils import remove_prefix
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import yaml


@dataclass
class YamlGenerator:
    """
    Class for generating the YAML config based on the given YAML schema.

    Attributes
    ----------
    schema: Dict
        The schema of the template.
    """

    schema: Dict[str, Any] = None

    def generate_example_dict(
        self,
        values: Optional[Dict[str, Any]] = (),
        required_keys: Optional[List[str]] = (),
    ) -> Dict:
        """
        Generate the YAML config based on the YAML schema.

        Properties
        ----------
        values: Optional dictionary containing specific values for the attributes.

        Returns
        -------
        Dict
            The generated dictionary config.
        """
        return self._generate_example(self.schema, values, required_keys)

    def generate_example(self, values: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate the YAML config based on the YAML schema.

        Properties
        ----------
        values: Optional dictionary containing specific values for the attributes.

        Returns
        -------
        str
            The generated YAML config.
        """
        return yaml.dump(self._generate_example(self.schema, values))

    def _check_condition(
        self, condition: Dict[str, Any], example: Dict[str, Any]
    ) -> bool:
        """
        Checks if the YAML schema condition fulfils.
        This method is used to include conditional fields into the final config.

        Properties
        ----------
        condition: Dict[str, Any]
            The schema condition.
            Example:
            In the example below the `owner_name` field has dependency on the `model` field.
            The `owner_name` will be included to the final config if only `model` is `prophet`.
                owner_name:
                    type: string
                    dependencies: {"model":"prophet"}
        example: Dict[str, Any]
            The config to check if the dependable value presented there.
        Returns
        -------
        bool
            True if the condition fulfills, false otherwise.
        """
        for key, value in condition.items():
            if key not in example or example[key] != value:
                return False
        return True

    def _generate_example(
        self,
        schema: Dict[str, Any],
        values: Optional[Dict[str, Any]] = (),
        required_keys: Optional[List[str]] = (),
    ) -> Dict[str, Any]:
        """
        Generates the final YAML config.
        This is a recursive method, which iterates through the entire schema.

        Properties
        ----------
        schema: Dict[str, Any]
            The schema to generate the config.
        values: Optional[Dict[str, Any]]
            The optional values that would be used instead of default values provided in the schema.

        Returns
        -------
        Dict
            The result config.
        """
        example = {}

        for key, value in schema.items():
            # only generate values for required fields
            if (
                value.get("required", False)
                or value.get("dependencies", False)
                or key in values
                or (required_keys and key in required_keys)
            ):
                if not "dependencies" in value or self._check_condition(
                    value["dependencies"], example
                ):
                    data_type = value.get("type")
                    if key in values:
                        example[key] = values[key]
                    elif "default" in value:
                        example[key] = value["default"]
                    elif data_type == "string":
                        example[key] = "value"
                    elif data_type == "number":
                        example[key] = 1
                    elif data_type == "boolean":
                        example[key] = True
                    elif data_type == "list":
                        # TODO: Handle list of dict
                        example[key] = ["item1", "item2"]
                    elif data_type == "dict":
                        example[key] = self._generate_example(
                            schema=value.get("schema", {}),
                            values={
                                remove_prefix(local_key, f"{key}."): value
                                for local_key, value in values.items()
                            },
                            required_keys=[
                                remove_prefix(required_key, f"{key}.")
                                for required_key in required_keys
                            ],
                        )

        return example
