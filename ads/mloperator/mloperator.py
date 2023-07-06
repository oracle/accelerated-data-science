#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import json
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

from ads.common.serializer import DataClassSerializable
from ads.mloperator.common.utils import OperatorValidator


@dataclass(repr=True)
class MLOperator(DataClassSerializable):
    kind: str = "operator"
    type: str = None
    name: str = None
    version: str = None
    spec: Any = None

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
        """
        schema = cls._load_schema()
        validator = OperatorValidator(schema)
        result = validator.validate(obj_dict)

        if not result:
            raise ValueError(
                "Invalid operator specification. Check the YAML structure and ensure it "
                "complies with the required schema for the operator. \n"
                f"{json.dumps(validator.errors, indent=2)}"
            )
        return True

    @classmethod
    @abstractmethod
    def _load_schema(cls) -> str:
        """Loads operator's schema."""
        raise NotImplementedError()
