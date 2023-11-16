#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import json
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

from ads.common.serializer import DataClassSerializable

from ads.opctl.operator.common.utils import OperatorValidator
from ads.opctl.operator.common.errors import OperatorSchemaYamlError


@dataclass(repr=True)
class OperatorConfig(DataClassSerializable):
    """Base class representing operator config.

    Attributes
    ----------
    kind: str
        The kind of the resource. For operators it is always - `operator`.
    type: str
        The type of the operator.
    version: str
        The version of the operator.
    spec: object
        The operator specification details.
    runtime: dict
        The runtime details of the operator.
    """

    kind: str = "operator"
    type: str = None
    version: str = None
    spec: Any = None
    runtime: Dict = None

    @classmethod
    def _validate_dict(cls, obj_dict: Dict) -> bool:
        """Validates the operator specification.

        Parameters
        ----------
        obj_dict: (dict)
            Dictionary representation of the object

        Returns
        -------
        bool
            True if the validation passed, else False.

        Raises
        ------
        ForecastSchemaYamlError
            In case of wrong specification format.
        """
        schema = cls._load_schema()
        validator = OperatorValidator(schema)
        validator.allow_unknown = True
        result = validator.validate(obj_dict)

        if not result:
            raise OperatorSchemaYamlError(json.dumps(validator.errors, indent=2))
        return True

    @classmethod
    @abstractmethod
    def _load_schema(cls) -> str:
        """
        The abstract method to load operator schema.
        This method needs to be implemented on the child level.
        Every operator will have their own YAML schema.
        """
        raise NotImplementedError()
