#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from dataclasses import dataclass, field

from ads.common.serializer import DataClassSerializable
from ads.opctl.operator.common.utils import _load_yaml_from_uri
from ads.opctl.operator.common.operator_config import OperatorConfig


@dataclass(repr=True)
class FeatureStoreOperatorSpec(DataClassSerializable):
    """Class representing the feature store operator specification."""

    operation: str = None
    stack_id: str = None
    service_version: str = None
    tenancy_id: str = None
    compartment_id: str = None
    region: str = None
    user_id = None


@dataclass(repr=True)
class FeatureStoreOperatorConfig(OperatorConfig):
    """Class representing the operator config.

    Attributes
    ----------
    kind: str
        The kind of the resource. For operators it is always - `operator`.
    type: str
        The type of the operator.
    version: str
        The version of the operator.
    spec: FeatureStoreOperatorSpec
        The operator specification.
    """

    kind: str = "operator"
    type: str = "feature_store"
    version: str = "v1"
    spec: FeatureStoreOperatorSpec = field(default_factory=FeatureStoreOperatorSpec)

    @classmethod
    def _load_schema(cls) -> str:
        """Loads operator schema."""
        return _load_yaml_from_uri(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.yaml")
        )
