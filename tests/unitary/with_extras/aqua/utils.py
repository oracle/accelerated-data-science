#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from dataclasses import dataclass
from typing import Dict

import pandas as pd
from pydantic import BaseModel, PositiveInt, ValidationError


@dataclass(repr=False)
class MockData:
    """Used for testing serializing dataclass in handler."""

    id: str = ""
    name: str = ""


class HandlerTestDataset:
    MOCK_OCID = "ocid.datasciencemdoel.<ocid>"
    mock_valid_input = dict(
        evaluation_source_id="ocid1.datasciencemodel.oc1.iad.<OCID>",
        evaluation_name="test_evaluation_name",
        dataset_path="oci://dataset_bucket@namespace/prefix/dataset.jsonl",
        report_path="oci://report_bucket@namespace/prefix/",
        model_parameters=dict(max_token=500),
        shape_name="VM.Standard.E3.Flex",
        block_storage_size=1,
        experiment_name="test_experiment_name",
        memory_in_gbs=1,
        ocpus=1,
    )
    mock_invalid_input = dict(name="myvalue")
    mock_dataclass_obj = MockData(id="myid", name="myname")

    def mock_url(self, action):
        return f"{self.MOCK_OCID}/{action}"


class SupportMetricsFormat(BaseModel):
    """Format for supported evaluation metrics."""

    use_case: list
    key: str
    name: str
    description: str
    args: dict


class EvaluationConfigFormat(BaseModel):
    """Evaluation config format."""

    model_params: dict
    shape: Dict[str, dict]
    default: Dict[str, PositiveInt]


def check(conf_schema, conf):
    """Check if the format of the output dictionary is correct."""
    try:
        conf_schema(**conf)
        return True
    except ValidationError:
        return False
