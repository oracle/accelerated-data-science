#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from dataclasses import dataclass

import pytest
import yaml

from ads.opctl.operator.common.operator_config import OperatorConfig


class TestOperatorConfig:
    def test_operator_config(self):
        # Test valid operator config

        @dataclass(repr=True)
        class MyOperatorConfig(OperatorConfig):
            @classmethod
            def _load_schema(cls) -> str:
                return yaml.safe_load(
                    """
                kind:
                    required: true
                    type: string
                version:
                    required: true
                    type: string
                type:
                    required: true
                    type: string
                spec:
                    required: true
                    type: dict
                    schema:
                        foo:
                            required: false
                            type: string
                """
                )

        config = MyOperatorConfig.from_dict(
            {
                "kind": "operator",
                "type": "my-operator",
                "version": "v1",
                "spec": {"foo": "bar"},
            }
        )
        assert config.kind == "operator"
        assert config.type == "my-operator"
        assert config.version == "v1"
        assert config.spec == {"foo": "bar"}

        # Test invalid operator config
        @dataclass(repr=True)
        class InvalidOperatorConfig(OperatorConfig):
            @classmethod
            def _load_schema(cls) -> str:
                return yaml.safe_load(
                    """
                kind:
                    required: true
                    type: string
                version:
                    required: true
                    type: string
                    allowed:
                        - v1
                type:
                    required: true
                    type: string
                spec:
                    required: true
                    type: dict
                    schema:
                        foo:
                            required: true
                            type: string
                """
                )

        with pytest.raises(ValueError):
            InvalidOperatorConfig.from_dict(
                {
                    "kind": "operator",
                    "type": "invalid-operator",
                    "version": "v2",
                    "spec": {"foo1": 123},
                }
            )
