#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
import pytest
from unittest.mock import patch, MagicMock

import oci.data_science.models

from ads.aqua.common.entities import ContainerSpec
from ads.aqua.config.config import get_evaluation_service_config
from ads.aqua.app import AquaApp


class TestConfig:
    """Unit tests for AQUA common configurations."""

    def setup_class(cls):
        cls.curr_dir = os.path.dirname(os.path.abspath(__file__))
        cls.artifact_dir = os.path.join(cls.curr_dir, "test_data", "config")

    @patch("ads.aqua.config.config.get_container_config")
    def test_evaluation_service_config(self, mock_get_container_config):
        """Ensures that the common evaluation configuration can be successfully retrieved."""

        with open(
            os.path.join(
                self.artifact_dir, "evaluation_config_with_default_params.json"
            )
        ) as file:
            expected_result = {
                ContainerSpec.CONTAINER_SPEC: {"test_container": json.load(file)}
            }

        mock_get_container_config.return_value = expected_result

        test_result = get_evaluation_service_config(container="test_container")
        assert (
            test_result.to_dict()
            == expected_result[ContainerSpec.CONTAINER_SPEC]["test_container"]
        )

    @pytest.mark.parametrize(
        "custom_metadata",
        [
            {
                "category": "Other",
                "description": "test_desc",
                "key": "artifact_location",
                "value": "artifact_location",
            },
            {},
        ],
    )
    @pytest.mark.parametrize("verified_model", [True, False])
    @pytest.mark.parametrize("path_exists", [True, False])
    @patch("ads.aqua.app.load_config")
    def test_load_config(
        self, mock_load_config, custom_metadata, verified_model, path_exists
    ):
        mock_load_config.return_value = {"config_key": "config_value"}
        service_model_tag = (
            {"aqua_service_model": "aqua_service_model_id"} if verified_model else {}
        )

        self.app = AquaApp()

        model = {
            "id": "mock_id",
            "lifecycle_details": "mock_lifecycle_details",
            "lifecycle_state": "mock_lifecycle_state",
            "project_id": "mock_project_id",
            "freeform_tags": {
                **{
                    "OCI_AQUA": "",
                },
                **service_model_tag,
            },
            "custom_metadata_list": [
                oci.data_science.models.Metadata(**custom_metadata)
            ],
        }

        self.app.ds_client.get_model = MagicMock(
            return_value=oci.response.Response(
                status=200,
                request=MagicMock(),
                headers=MagicMock(),
                data=oci.data_science.models.Model(**model),
            )
        )
        with patch("ads.aqua.app.is_path_exists", return_value=path_exists):
            result = self.app.get_config(
                model_id="test_model_id", config_file_name="test_config_file_name"
            )
            if not path_exists:
                assert result == {}
            if not custom_metadata:
                assert result == {}
            if path_exists and custom_metadata:
                assert result == {"config_key": "config_value"}
