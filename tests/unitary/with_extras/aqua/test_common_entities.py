#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest

from ads.aqua.common.entities import ComputeShapeSummary


class TestComputeShapeSummary:
    @pytest.mark.parametrize(
        "input_data, expected_gpu_specs",
        [
            # Case 1: Shape is present in GPU_SPECS.
            (
                {
                    "core_count": 32,
                    "memory_in_gbs": 512,
                    "name": "VM.GPU2.1",
                    "shape_series": "GPU",
                },
                {"gpu_type": "P100", "gpu_count": 1, "gpu_memory_in_gbs": 16},
            ),
            # Case 2: Not in GPU_SPECS; fallback extraction should yield gpu_count.
            (
                {
                    "core_count": 16,
                    "memory_in_gbs": 256,
                    "name": "VM.GPU.UNKNOWN.4",
                    "shape_series": "GPU",
                },
                {"gpu_type": None, "gpu_count": 4, "gpu_memory_in_gbs": None},
            ),
            # Case 3: Non-GPU shape should not populate GPU specs.
            (
                {
                    "core_count": 8,
                    "memory_in_gbs": 64,
                    "name": "VM.Standard2.1",
                    "shape_series": "STANDARD",
                },
                None,
            ),
        ],
    )
    def test_set_gpu_specs(self, input_data, expected_gpu_specs):
        shape = ComputeShapeSummary(**input_data)
        if expected_gpu_specs is None:
            assert shape.gpu_specs is None
        else:
            assert shape.gpu_specs is not None
            # Verify GPU type, count, and memory.
            assert shape.gpu_specs.gpu_type == expected_gpu_specs.get("gpu_type")
            assert shape.gpu_specs.gpu_count == expected_gpu_specs.get("gpu_count")
            assert shape.gpu_specs.gpu_memory_in_gbs == expected_gpu_specs.get(
                "gpu_memory_in_gbs"
            )
