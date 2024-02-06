#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from dataclasses import dataclass
from typing import ClassVar, Dict

from ads.opctl.operator.common.utils import _load_yaml_from_uri

from ads.opctl.operator.common.operator_yaml_generator import YamlGenerator

from ads.common.serializer import DataClassSerializable

from ads.common.extended_enum import ExtendedEnum

from ads.opctl.operator.runtime.runtime import Runtime


class OPERATOR_MARKETPLACE_LOCAL_RUNTIME_TYPE(ExtendedEnum):
    PYTHON = "python"


MARKETPLACE_OPERATOR_LOCAL_KIND = "marketplace.local"


@dataclass(repr=True)
class MarketplacePythonRuntime(Runtime):
    """Represents a python operator runtime."""

    _schema: ClassVar[str] = "python_marketplace_runtime_schema.yaml"
    type: str = OPERATOR_MARKETPLACE_LOCAL_RUNTIME_TYPE.PYTHON.value
    version: str = "v1"

    def __init__(self, **kwargs):
        kwargs.update(kind=MARKETPLACE_OPERATOR_LOCAL_KIND)
        self.spec = YamlGenerator(
            schema=_load_yaml_from_uri(
                __file__.replace("marketplace_runtime.py", self._schema)
            )
        ).generate_example_dict()

        super().__init__(**kwargs)

    @classmethod
    def init(cls, **kwargs: Dict) -> "MarketplacePythonRuntime":
        """Initializes a starter specification for the runtime.

        Returns
        -------
        PythonRuntime
            The runtime instance.
        """
        instance = cls()
        return instance
