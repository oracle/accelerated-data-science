#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import typing
from dataclasses import dataclass, fields
from typing import Dict


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
    mock_service_payload_create = {
        "target_service": "data_science",
        "status": 404,
        "code": "NotAuthenticated",
        "opc-request-id": "1234",
        "message": "The required information to complete authentication was not provided or was incorrect.",
        "operation_name": "create_resources",
        "timestamp": "2024-04-12T02:51:24.977404+00:00",
        "request_endpoint": "POST xxx",
    }
    mock_service_payload_get = {
        "target_service": "data_science",
        "status": 404,
        "code": "NotAuthenticated",
        "opc-request-id": "1234",
        "message": "The required information to complete authentication was not provided or was incorrect.",
        "operation_name": "get_job_run",
        "timestamp": "2024-04-12T02:51:24.977404+00:00",
        "request_endpoint": "GET xxx",
    }

    def mock_url(self, action):
        return f"{self.MOCK_OCID}/{action}"


@dataclass
class BaseFormat:
    """Implements type checking for each format."""

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            field_type = (
                field.type.__origin__
                if isinstance(field.type, typing._GenericAlias)
                else field.type
            )
            if not isinstance(value, field_type):
                raise TypeError(
                    f"Expected {field.name} to be {field_type}, " f"got {repr(value)}"
                )


@dataclass
class SupportMetricsFormat(BaseFormat):
    """Format for supported evaluation metrics."""

    use_case: list
    key: str
    name: str
    description: str
    args: dict


@dataclass
class EvaluationConfigFormat(BaseFormat):
    """Evaluation config format."""

    model_params: dict
    shape: Dict[str, dict]
    default: Dict[str, int]


def check(conf_schema, conf):
    """Check if the format of the output dictionary is correct."""
    try:
        conf_schema(**conf)
        return True
    except TypeError:
        return False
