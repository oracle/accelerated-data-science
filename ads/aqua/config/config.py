#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


# TODO: move this to global config.json in object storage
def get_finetuning_config_defaults():
    """Generate and return the fine-tuning default configuration dictionary."""
    return {
        "shape": {
            "VM.GPU.A10.1": {"batch_size": 1, "replica": "1-10"},
            "VM.GPU.A10.2": {"batch_size": 1, "replica": "1-10"},
            "BM.GPU.A10.4": {"batch_size": 1, "replica": 1},
            "BM.GPU4.8": {"batch_size": 4, "replica": 1},
            "BM.GPU.A100-v2.8": {"batch_size": 6, "replica": 1},
        }
    }
