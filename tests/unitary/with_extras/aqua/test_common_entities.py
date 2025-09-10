#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict, List, Union
from unittest.mock import MagicMock, patch

import pytest

from ads.aqua.common.entities import (
    AquaMultiModelRef,
    ComputeShapeSummary,
    ContainerPath,
)


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
                    "gpu_specs": {
                        "gpu_type": "P100",
                        "gpu_count": 1,
                        "gpu_memory_in_gbs": 16,
                    },
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


class TestContainerPath:
    """The unit tests for ContainerPath."""

    @pytest.mark.parametrize(
        "image_path, expected_result",
        [
            (
                "iad.ocir.io/ociodscdev/odsc-llm-evaluate:0.1.2.9",
                {
                    "full_path": "iad.ocir.io/ociodscdev/odsc-llm-evaluate:0.1.2.9",
                    "path": "iad.ocir.io/ociodscdev/odsc-llm-evaluate",
                    "name": "odsc-llm-evaluate",
                    "version": "0.1.2.9",
                },
            ),
            (
                "dsmc://model-with-version:0.2.78.0",
                {
                    "full_path": "dsmc://model-with-version:0.2.78.0",
                    "path": "dsmc://model-with-version",
                    "name": "model-with-version",
                    "version": "0.2.78.0",
                },
            ),
            (
                "oci://my-custom-model-version:1.0.0",
                {
                    "full_path": "oci://my-custom-model-version:1.0.0",
                    "path": "oci://my-custom-model-version",
                    "name": "my-custom-model-version",
                    "version": "1.0.0",
                },
            ),
            (
                "custom-scheme://path/to/versioned-model:2.5.1",
                {
                    "full_path": "custom-scheme://path/to/versioned-model:2.5.1",
                    "path": "custom-scheme://path/to/versioned-model",
                    "name": "versioned-model",
                    "version": "2.5.1",
                },
            ),
            (
                "custom-scheme://path/to/versioned-model",
                {
                    "full_path": "custom-scheme://path/to/versioned-model",
                    "path": "custom-scheme://path/to/versioned-model",
                    "name": "versioned-model",
                    "version": None,
                },
            ),
        ],
    )
    def test_positive(self, image_path, expected_result):
        assert ContainerPath(full_path=image_path).model_dump() == expected_result


class TestAquaMultiModelRef:
    @pytest.mark.parametrize(
        "env_var, params, expected_params",
        [
            (
                {"PARAMS": "--max-model-len 8192 --enforce-eager"},
                {},
                {"--max-model-len": "8192", "--enforce-eager": "UNKNOWN"},
            ),
            (
                {"PARAMS": "--a 1 --b 2"},
                {"--a": "existing"},
                {"--a": "existing", "--b": "2"},
            ),
            (
                {"PARAMS": "--x 1"},
                None,
                {"--x": "1"},
            ),
            (
                {},  # No PARAMS key
                {"--existing": "value"},
                {"--existing": "value"},
            ),
        ],
    )
    @patch.object(AquaMultiModelRef, "_parse_params")
    def test_extract_params_from_env_var(
        self, mock_parse_params, env_var, params, expected_params
    ):
        mock_parse_params.return_value = {k: v for k, v in expected_params.items()}

        values = {
            "model_id": "ocid1.model.oc1..xxxx",
            "env_var": dict(env_var),  # copy
            "params": params,
        }

        result = AquaMultiModelRef.model_validate(values)
        assert result.params == expected_params
        assert "PARAMS" not in result.env_var

    @patch.object(AquaMultiModelRef, "_parse_params")
    def test_extract_params_from_env_var_skips_override(self, mock_parse_params):
        input_params = {"--max-model-len": "65536"}
        env_var = {"PARAMS": "--max-model-len 8000 --new-flag yes"}

        mock_parse_params.return_value = {
            "--max-model-len": "8000",
            "--new-flag": "yes",
        }

        values = {
            "model_id": "ocid1.model.oc1..abcd",
            "params": dict(input_params),
            "env_var": dict(env_var),
        }

        result = AquaMultiModelRef.model_validate(values)
        assert result.params["--max-model-len"] == "65536"  # original
        assert result.params["--new-flag"] == "yes"

    def test_extract_params_from_env_var_missing_env(self):
        values = {
            "model_id": "ocid1.model.oc1..abcd",
        }
        result = AquaMultiModelRef.model_validate(values)
        assert result.env_var == {}
        assert result.params == {}

    def test_all_model_ids_no_finetunes(self):
        model = AquaMultiModelRef(model_id="ocid1.model.oc1..base")
        assert model.all_model_ids() == ["ocid1.model.oc1..base"]

    @patch.object(AquaMultiModelRef, "_parse_params")
    def test_model_validator_with_other_fields(self, mock_parse_params):
        values = {
            "model_id": "ocid1.model.oc1..xyz",
            "gpu_count": 2,
            "artifact_location": "some/path",
            "env_var": {"PARAMS": "--x abc"},
        }

        mock_parse_params.return_value = {"--x": "abc"}

        result = AquaMultiModelRef.model_validate(values)

        assert result.model_id == "ocid1.model.oc1..xyz"
        assert result.gpu_count == 2
        assert result.artifact_location == "some/path"
        assert result.params == {"--x": "abc"}

    @pytest.mark.parametrize(
        "input_param,expected_dict",
        [
            (
                "--max-model-len 65536 --enable-streaming",
                {"--max-model-len": "65536", "--enable-streaming": ""},
            ),
            (
                ["--max-model-len 4096", "--foo bar"],
                {"--max-model-len": "4096", "--foo": "bar"},
            ),
            (
                "",
                {},
            ),
            (
                None,
                {},
            ),
            (
                "--key1 value1 --key2 value with spaces",
                {"--key1": "value1", "--key2": "value with spaces"},
            ),
        ],
    )
    def test_parse_params(
        self, input_param: Union[str, List[str]], expected_dict: Dict[str, str]
    ):
        result = AquaMultiModelRef._parse_params(input_param)
        assert result == expected_dict
