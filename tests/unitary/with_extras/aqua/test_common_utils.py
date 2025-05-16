#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest
from ads.aqua.common.utils import get_preferred_compatible_family


class TestCommonUtils:
    @pytest.mark.parametrize(
        "input_families, expected",
        [
            (
                {"odsc-vllm-serving", "odsc-vllm-serving-v1"},
                "odsc-vllm-serving-v1",
            ),
            (
                {"odsc-vllm-serving", "odsc-vllm-serving-llama4"},
                "odsc-vllm-serving-llama4",
            ),
            (
                {"odsc-vllm-serving-v1", "odsc-vllm-serving-llama4"},
                "odsc-vllm-serving-llama4",
            ),
            (
                {
                    "odsc-vllm-serving",
                    "odsc-vllm-serving-v1",
                    "odsc-vllm-serving-llama4",
                },
                "odsc-vllm-serving-llama4",
            ),
            ({"odsc-tgi-serving", "odsc-vllm-serving"}, None),
            ({"non-existing-one", "odsc-tgi-serving"}, None),
            ({"odsc-tgi-serving", "odsc-vllm-serving-llama4"}, None),
            ({"odsc-tgi-serving", "odsc-vllm-serving-v1"}, None),
        ],
    )
    def test_get_preferred_compatible_family(self, input_families, expected):
        assert get_preferred_compatible_family(input_families) == expected
