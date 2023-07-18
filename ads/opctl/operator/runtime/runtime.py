#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import json
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List

from cerberus import Validator

from ads.common.serializer import DataClassSerializable
from ads.opctl.operator.common.utils import _load_yaml_from_uri


@dataclass(repr=True)
class Runtime(DataClassSerializable):
    """Base class for the operator's runtimes."""

    _schema: ClassVar[str] = None
    kind: str = None
    type: str = None
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
        schema = _load_yaml_from_uri(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), cls._schema)
        )
        validator = Validator(schema, purge_unknown=True)
        result = validator.validate(obj_dict)

        if not result:
            raise ValueError(
                "Invalid runtime specification. Check the YAML structure and ensure it "
                "complies with the required schema for the runtime. \n"
                f"{json.dumps(validator.errors, indent=2)}"
            )
        return True


@dataclass(repr=True)
class ContainerRuntimeSpec(DataClassSerializable):
    """Represents a container operator runtime specification."""

    image: str = None
    env: List[Dict] = field(default_factory=list)
    volume: List[str] = field(default_factory=list)


@dataclass(repr=True)
class ContainerRuntime(Runtime):
    """Represents a container operator runtime."""

    _schema: ClassVar[str] = "container_runtime_schema.yaml"
    kind: str = "operator.local"
    type: str = "container"
    version: str = "v1"
    spec: ContainerRuntimeSpec = field(default_factory=ContainerRuntimeSpec)

    @classmethod
    def init(cls, **kwargs: Dict) -> "ContainerRuntime":
        """Initializes a starter specification for the runtime.

        Returns
        -------
        ContainerRuntime
            The runtime instance.
        """
        return cls(spec=ContainerRuntimeSpec.from_dict(kwargs))
