#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

GPU_SPECS = {
    "VM.GPU2.1": {
        "gpu_type": "P100",
        "gpu_count": 1,
        "gpu_memory_in_gbs": 16,
    },
    "VM.GPU3.1": {
        "gpu_type": "V100",
        "gpu_count": 1,
        "gpu_memory_in_gbs": 16,
    },
    "VM.GPU3.2": {
        "gpu_type": "V100",
        "gpu_count": 2,
        "gpu_memory_in_gbs": 32,
    },
    "VM.GPU3.4": {
        "gpu_type": "V100",
        "gpu_count": 4,
        "gpu_memory_in_gbs": 64,
    },
    "BM.GPU2.2": {
        "gpu_type": "P100",
        "gpu_count": 2,
        "gpu_memory_in_gbs": 32,
    },
    "BM.GPU3.8": {
        "gpu_type": "V100",
        "gpu_count": 8,
        "gpu_memory_in_gbs": 128,
    },
    "BM.GPU4.8": {
        "gpu_type": "A100",
        "gpu_count": 8,
        "gpu_memory_in_gbs": 320,
    },
    "BM.GPU.A10.4": {
        "gpu_type": "A10",
        "gpu_count": 4,
        "gpu_memory_in_gbs": 96,
    },
    "VM.GPU.A10.4": {
        "gpu_type": "A10",
        "gpu_count": 4,
        "gpu_memory_in_gbs": 96,
    },
    "BM.GPU.H100.8": {
        "gpu_type": "H100",
        "gpu_count": 8,
        "gpu_memory_in_gbs": 640,
    },
    "VM.GPU.A10.1": {
        "gpu_type": "A10",
        "gpu_count": 1,
        "gpu_memory_in_gbs": 24,
    },
    "VM.GPU.A10.2": {
        "gpu_type": "A10",
        "gpu_count": 2,
        "gpu_memory_in_gbs": 48,
    },
    "BM.GPU.L40S-NC.4": {
        "gpu_type": "L40S",
        "gpu_count": 4,
        "gpu_memory_in_gbs": 192,
    },
    "BM.GPU.H200.8": {
        "gpu_type": "H200",
        "gpu_count": 8,
        "gpu_memory_in_gbs": 1128,
    },
    "BM.GPU.A100-v2.8": {
        "gpu_type": "A100",
        "gpu_count": 8,
        "gpu_memory_in_gbs": 320,
    },
}
