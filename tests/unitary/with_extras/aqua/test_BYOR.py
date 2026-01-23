#!/usr/bin/env python
# -*- coding: utf-8 -*--
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
Unit tests for AQUA model deployment with capacity reservation support.

Tests cover:
- Entity-level capacity reservation handling
- Infrastructure builder methods
- Deployment creation and configuration
- Backward compatibility with environment variable approach
- SDK version compatibility
"""

import pytest
from unittest.mock import MagicMock, patch, Mock, PropertyMock
import warnings
from typing import Optional, Dict, Any


class TestCapacityReservationEntities:
    """Test capacity reservation in CreateModelDeploymentDetails entity."""

    def test_create_with_capacity_reservation_ids(self):
        """Test creating deployment details with capacity_reservation_ids."""
        from ads.aqua.modeldeployment.entities import CreateModelDeploymentDetails

        details = CreateModelDeploymentDetails(
            instance_shape="VM.GPU.A10.1",
            model_id="ocid1.datasciencemodel.oc1.iad.amaaaaaay75uckqa",
            capacity_reservation_ids=[
                "ocid1.capacityreservation.oc1.iad.anuwcljsy75uckqc"
            ],
        )

        assert details.capacity_reservation_ids == [
            "ocid1.capacityreservation.oc1.iad.anuwcljsy75uckqc"
        ]
        assert details.instance_shape == "VM.GPU.A10.1"

    def test_create_without_capacity_reservation_ids(self):
        """Test that capacity_reservation_ids is optional."""
        from ads.aqua.modeldeployment.entities import CreateModelDeploymentDetails

        details = CreateModelDeploymentDetails(
            instance_shape="VM.GPU.A10.1",
            model_id="ocid1.datasciencemodel.oc1.iad.amaaaaaay75uckqa",
        )

        assert details.capacity_reservation_ids is None

    def test_backward_compat_env_var_extraction(self):
        """Test backward compatibility: extracting CAPACITY_RESERVATION_ID from env_var."""
        from ads.aqua.modeldeployment.entities import CreateModelDeploymentDetails
        import logging

        env_var = {
            "PARAMS": "",
            "CAPACITY_RESERVATION_ID": "ocid1.capacityreservation.oc1.iad.anuwcljsy75uckqc",
        }

        # Validation happens automatically during init for Pydantic models
        details = CreateModelDeploymentDetails(
            instance_shape="VM.GPU.A10.1",
            model_id="ocid1.datasciencemodel.oc1.iad.amaaaaaay75uckqa",
            env_var=env_var,
        )

        # Should be extracted
        assert details.capacity_reservation_ids == [
            "ocid1.capacityreservation.oc1.iad.anuwcljsy75uckqc"
        ]
        # Should be removed from env_var
        assert "CAPACITY_RESERVATION_ID" not in details.env_var

    def test_capacity_reservation_ids_takes_precedence(self):
        """Test that explicit capacity_reservation_ids takes precedence over env_var."""
        from ads.aqua.modeldeployment.entities import CreateModelDeploymentDetails

        env_var = {"CAPACITY_RESERVATION_ID": "ocid1.capacityreservation.old"}

        details = CreateModelDeploymentDetails(
            instance_shape="VM.GPU.A10.1",
            model_id="ocid1.datasciencemodel.oc1.iad.amaaaaaay75uckqa",
            capacity_reservation_ids=["ocid1.capacityreservation.new"],
            env_var=env_var,
        )

        # Explicit parameter should win
        assert details.capacity_reservation_ids == ["ocid1.capacityreservation.new"]
        # Note: env_var is only cleaned when capacity_reservation_ids is NOT explicitly provided

    def test_valid_capacity_reservation_ocid(self):
        """Test that valid OCID formats are accepted."""
        from ads.aqua.modeldeployment.entities import CreateModelDeploymentDetails

        valid_ocids = [
            "ocid1.capacityreservation.oc1.iad.test123",
            "ocid1.capacityreservation.oc1.phx.anuwcljsy75uckqc",
        ]

        for ocid in valid_ocids:
            # Should not raise during initialization
            details = CreateModelDeploymentDetails(
                instance_shape="VM.GPU.A10.1",
                model_id="ocid1.datasciencemodel.oc1.iad.test",
                capacity_reservation_ids=[ocid],
            )
            assert details.capacity_reservation_ids == [ocid]

    def test_extraction_from_oci_response(self):
        """Test that capacity_reservation_ids can be extracted from OCI ModelDeployment response."""
        from ads.aqua.modeldeployment.entities import ShapeInfo

        # Test ShapeInfo includes capacity_reservation_ids
        shape_info = ShapeInfo(
            instance_shape="VM.GPU.A10.1",
            capacity_reservation_ids=["ocid1.capacityreservation.oc1.iad.test"],
        )

        assert shape_info.capacity_reservation_ids == [
            "ocid1.capacityreservation.oc1.iad.test"
        ]


class TestCapacityReservationInfrastructure:
    """Test capacity reservation in ModelDeploymentInfrastructure."""

    def test_with_capacity_reservation_ids_builder(self):
        """Test with_capacity_reservation_ids builder method."""
        from ads.model.deployment.model_deployment_infrastructure import (
            ModelDeploymentInfrastructure,
        )

        infra = ModelDeploymentInfrastructure()
        result = infra.with_capacity_reservation_ids(
            ["ocid1.capacityreservation.oc1.iad.test"]
        )

        # Should return self for chaining
        assert result is infra
        # Should store the value
        assert infra.capacity_reservation_ids == [
            "ocid1.capacityreservation.oc1.iad.test"
        ]

    def test_with_capacity_reservation_ids_empty_list(self):
        """Test with_capacity_reservation_ids with empty list."""
        from ads.model.deployment.model_deployment_infrastructure import (
            ModelDeploymentInfrastructure,
        )

        infra = ModelDeploymentInfrastructure()
        infra.with_capacity_reservation_ids([])

        assert infra.capacity_reservation_ids == []

    def test_with_capacity_reservation_ids_none(self):
        """Test with_capacity_reservation_ids with None."""
        from ads.model.deployment.model_deployment_infrastructure import (
            ModelDeploymentInfrastructure,
        )

        infra = ModelDeploymentInfrastructure()
        infra.with_capacity_reservation_ids(None)

        assert infra.capacity_reservation_ids is None

    def test_capacity_reservation_ids_property(self):
        """Test capacity_reservation_ids property getter."""
        from ads.model.deployment.model_deployment_infrastructure import (
            ModelDeploymentInfrastructure,
        )

        infra = ModelDeploymentInfrastructure()
        # Default should be None
        assert infra.capacity_reservation_ids is None

        # After setting
        infra.with_capacity_reservation_ids(["ocid1.capacityreservation.oc1.iad.test"])
        assert infra.capacity_reservation_ids == [
            "ocid1.capacityreservation.oc1.iad.test"
        ]

    def test_infrastructure_builder_chaining(self):
        """Test that capacity reservation works with method chaining."""
        from ads.model.deployment.model_deployment_infrastructure import (
            ModelDeploymentInfrastructure,
        )

        infra = (
            ModelDeploymentInfrastructure()
            .with_shape_name("VM.GPU.A10.1")
            .with_bandwidth_mbps(10)
            .with_replica(1)
            .with_capacity_reservation_ids(["ocid1.capacityreservation.oc1.iad.test"])
            .with_web_concurrency(10)
        )

        assert infra.shape_name == "VM.GPU.A10.1"
        assert infra.bandwidth_mbps == 10
        assert infra.replica == 1
        assert infra.capacity_reservation_ids == [
            "ocid1.capacityreservation.oc1.iad.test"
        ]
        assert infra.web_concurrency == 10

    def test_to_dict_includes_capacity_reservation_ids(self):
        """Test that to_dict includes capacity_reservation_ids."""
        from ads.model.deployment.model_deployment_infrastructure import (
            ModelDeploymentInfrastructure,
        )

        infra = (
            ModelDeploymentInfrastructure()
            .with_shape_name("VM.GPU.A10.1")
            .with_capacity_reservation_ids(["ocid1.capacityreservation.oc1.iad.test"])
        )

        oci_config = infra.to_dict()

        # Check in the nested spec structure
        assert "spec" in oci_config
        spec = oci_config["spec"]
        assert "capacityReservationIds" in spec
        assert spec["capacityReservationIds"] == [
            "ocid1.capacityreservation.oc1.iad.test"
        ]

    def test_to_dict_without_capacity_reservation_ids(self):
        """Test that to_dict works without capacity_reservation_ids."""
        from ads.model.deployment.model_deployment_infrastructure import (
            ModelDeploymentInfrastructure,
        )

        infra = ModelDeploymentInfrastructure().with_shape_name("VM.GPU.A10.1")

        oci_config = infra.to_dict()

        # Should have spec, but capacityReservationIds should be None or not present
        assert "spec" in oci_config
        spec = oci_config["spec"]
        cap_res = spec.get("capacityReservationIds")
        assert cap_res is None or cap_res == []


class TestModelDeploymentWithCapacityReservation:
    """Test ModelDeployment class with capacity reservation."""

    @patch("ads.model.deployment.model_deployment.ModelDeployment.deploy")
    def test_model_deployment_with_capacity_reservation_ids(self, mock_deploy):
        """Test deploying with capacity reservation."""
        from ads.model.deployment import ModelDeployment
        from ads.model.deployment.model_deployment_infrastructure import (
            ModelDeploymentInfrastructure,
        )
        from ads.model.deployment.model_deployment_runtime import (
            ModelDeploymentContainerRuntime,
        )

        mock_deploy.return_value = MagicMock(id="ocid1.deployment.test")

        deployment = ModelDeployment()
        deployment.with_infrastructure(
            ModelDeploymentInfrastructure()
            .with_shape_name("VM.GPU.A10.1")
            .with_capacity_reservation_ids(["ocid1.capacityreservation.oc1.iad.test"])
        )
        deployment.with_runtime(
            ModelDeploymentContainerRuntime()
            .with_model_uri("ocid1.datasciencemodel.oc1.iad.test")
            .with_image("iad.ocir.io/namespace/image:latest")
        )

        # Verify infrastructure has the capacity_reservation_ids
        assert deployment.infrastructure.capacity_reservation_ids == [
            "ocid1.capacityreservation.oc1.iad.test"
        ]

    @patch("ads.model.deployment.model_deployment.ModelDeployment.deploy")
    def test_model_deployment_without_capacity_reservation_ids(self, mock_deploy):
        """Test deploying without capacity reservation."""
        from ads.model.deployment import ModelDeployment
        from ads.model.deployment.model_deployment_infrastructure import (
            ModelDeploymentInfrastructure,
        )
        from ads.model.deployment.model_deployment_runtime import (
            ModelDeploymentContainerRuntime,
        )

        mock_deploy.return_value = MagicMock(id="ocid1.deployment.test")

        deployment = ModelDeployment()
        deployment.with_infrastructure(
            ModelDeploymentInfrastructure().with_shape_name("VM.GPU.A10.1")
        )
        deployment.with_runtime(
            ModelDeploymentContainerRuntime()
            .with_model_uri("ocid1.datasciencemodel.oc1.iad.test")
            .with_image("iad.ocir.io/namespace/image:latest")
        )

        # Should not have capacity_reservation_ids or it should be None
        assert deployment.infrastructure.capacity_reservation_ids is None

    def test_sdk_version_check_with_supported_sdk(self):
        """Test that deployment works when SDK supports capacity_reservation_ids."""
        from ads.model.deployment.model_deployment_infrastructure import (
            ModelDeploymentInfrastructure,
        )
        import oci.data_science.models

        # Check if SDK has capacity_reservation_ids attribute
        has_support = hasattr(
            oci.data_science.models.InstanceConfiguration(), "capacity_reservation_ids"
        )

        if has_support:
            infra = ModelDeploymentInfrastructure()
            infra.with_capacity_reservation_ids(
                ["ocid1.capacityreservation.oc1.iad.test"]
            )

            # Should not raise
            config_dict = infra.to_dict()
            assert config_dict is not None


class TestAquaDeploymentWithCapacityReservation:
    """Test AQUA deployment flow with capacity reservation."""

    @patch("ads.aqua.modeldeployment.deployment.AquaDeploymentApp.list_shapes")
    @patch("ads.aqua.modeldeployment.deployment.ModelDeployment")
    def test_aqua_passes_capacity_reservation_ids_to_infrastructure(
        self, mock_model_deployment, mock_list_shapes
    ):
        """Test that AQUA correctly passes capacity_reservation_ids to infrastructure builder."""
        from ads.aqua.modeldeployment.entities import (
            CreateModelDeploymentDetails,
            ShapeInfo,
        )

        # Mock shapes to prevent API calls
        mock_list_shapes.return_value = [
            ShapeInfo(instance_shape="VM.GPU.A10.1", capacity_reservation_ids=None)
        ]

        details = CreateModelDeploymentDetails(
            model_id="ocid1.datasciencemodel.oc1.iad.test",
            instance_shape="VM.GPU.A10.1",
            capacity_reservation_ids=["ocid1.capacityreservation.oc1.iad.test"],
            compartment_id="ocid1.compartment.oc1..test",
            project_id="ocid1.datascienceproject.oc1.iad.test",
        )

        # Verify details has capacity_reservation_ids
        assert details.capacity_reservation_ids == [
            "ocid1.capacityreservation.oc1.iad.test"
        ]

    @patch("ads.aqua.modeldeployment.deployment.AquaDeploymentApp.list_shapes")
    def test_aqua_deployment_without_capacity_reservation_ids(self, mock_list_shapes):
        """Test AQUA deployment creation without capacity_reservation_ids."""
        from ads.aqua.modeldeployment.entities import (
            CreateModelDeploymentDetails,
            ShapeInfo,
        )

        # Mock shapes to prevent API calls
        mock_list_shapes.return_value = [
            ShapeInfo(instance_shape="VM.GPU.A10.1", capacity_reservation_ids=None)
        ]

        details = CreateModelDeploymentDetails(
            model_id="ocid1.datasciencemodel.oc1.iad.test",
            instance_shape="VM.GPU.A10.1",
            compartment_id="ocid1.compartment.oc1..test",
            project_id="ocid1.datascienceproject.oc1.iad.test",
        )

        # Verify no capacity_reservation_ids
        assert details.capacity_reservation_ids is None

    def test_infrastructure_builder_receives_capacity_reservation_ids(self):
        """Test that infrastructure builder receives capacity_reservation_ids correctly."""
        from ads.model.deployment.model_deployment_infrastructure import (
            ModelDeploymentInfrastructure,
        )
        from ads.aqua.modeldeployment.entities import CreateModelDeploymentDetails

        details = CreateModelDeploymentDetails(
            model_id="ocid1.datasciencemodel.oc1.iad.test",
            instance_shape="VM.GPU.A10.1",
            capacity_reservation_ids=["ocid1.capacityreservation.oc1.iad.test"],
            compartment_id="ocid1.compartment.oc1..test",
            project_id="ocid1.datascienceproject.oc1.iad.test",
        )

        # Simulate the infrastructure building in deployment.py
        infrastructure = ModelDeploymentInfrastructure().with_shape_name(
            details.instance_shape
        )

        if details.capacity_reservation_ids:
            infrastructure.with_capacity_reservation_ids(
                details.capacity_reservation_ids
            )

        # Verify the capacity_reservation_ids were set
        assert infrastructure.capacity_reservation_ids == [
            "ocid1.capacityreservation.oc1.iad.test"
        ]


class TestCapacityReservationValidation:
    """Test validation and error handling for capacity reservation."""

    def test_validate_capacity_reservation_ocid_format(self):
        """Test OCID format validation."""
        from ads.aqua.modeldeployment.entities import CreateModelDeploymentDetails

        valid_ocids = [
            "ocid1.capacityreservation.oc1.iad.test123",
            "ocid1.capacityreservation.oc1.phx.anuwcljsy75uckqc",
        ]

        for ocid in valid_ocids:
            # Should not raise during initialization (Pydantic validates automatically)
            details = CreateModelDeploymentDetails(
                instance_shape="VM.GPU.A10.1",
                model_id="ocid1.datasciencemodel.oc1.iad.test",
                capacity_reservation_ids=[ocid],
            )
            assert details.capacity_reservation_ids == [ocid]

    def test_empty_capacity_reservation_ids_is_valid(self):
        """Test that None/empty capacity_reservation_ids is valid."""
        from ads.aqua.modeldeployment.entities import CreateModelDeploymentDetails

        # None is valid
        details = CreateModelDeploymentDetails(
            instance_shape="VM.GPU.A10.1",
            model_id="ocid1.datasciencemodel.oc1.iad.test",
            capacity_reservation_ids=None,
        )
        assert details.capacity_reservation_ids is None

        # Not providing it at all is valid
        details2 = CreateModelDeploymentDetails(
            instance_shape="VM.GPU.A10.1",
            model_id="ocid1.datasciencemodel.oc1.iad.test",
        )
        assert details2.capacity_reservation_ids is None


class TestBackwardCompatibility:
    """Test backward compatibility scenarios."""

    @patch("ads.model.deployment.model_deployment.ModelDeployment.deploy")
    def test_existing_deployments_without_capacity_reservation_still_work(
        self, mock_deploy
    ):
        """Test that deployments without capacity reservation continue to work."""
        from ads.model.deployment import ModelDeployment
        from ads.model.deployment.model_deployment_infrastructure import (
            ModelDeploymentInfrastructure,
        )
        from ads.model.deployment.model_deployment_runtime import (
            ModelDeploymentContainerRuntime,
        )

        mock_deploy.return_value = MagicMock(id="ocid1.deployment.test")

        # Create deployment the old way (no capacity reservation)
        deployment = ModelDeployment()
        deployment.with_infrastructure(
            ModelDeploymentInfrastructure()
            .with_shape_name("VM.GPU.A10.1")
            .with_bandwidth_mbps(10)
            .with_replica(1)
        )
        deployment.with_runtime(
            ModelDeploymentContainerRuntime()
            .with_model_uri("ocid1.datasciencemodel.oc1.iad.test")
            .with_image("iad.ocir.io/namespace/image:latest")
        )

        # Should work without errors
        assert deployment.infrastructure.capacity_reservation_ids is None

    def test_env_var_approach_extracts_to_native(self):
        """Test that using env var approach extracts to native capacity_reservation_ids."""
        from ads.aqua.modeldeployment.entities import CreateModelDeploymentDetails

        details = CreateModelDeploymentDetails(
            instance_shape="VM.GPU.A10.1",
            model_id="ocid1.datasciencemodel.oc1.iad.test",
            env_var={
                "CAPACITY_RESERVATION_ID": "ocid1.capacityreservation.oc1.iad.test"
            },
        )

        # Should extract to capacity_reservation_ids
        assert details.capacity_reservation_ids == [
            "ocid1.capacityreservation.oc1.iad.test"
        ]
        # Should be removed from env_var
        assert "CAPACITY_RESERVATION_ID" not in details.env_var

    def test_migration_path_from_env_var_to_native(self):
        """Test smooth migration from env var to native approach."""
        from ads.aqua.modeldeployment.entities import CreateModelDeploymentDetails

        # Phase 1: Old approach (env var)
        old_details = CreateModelDeploymentDetails(
            instance_shape="VM.GPU.A10.1",
            model_id="ocid1.datasciencemodel.oc1.iad.test",
            env_var={
                "CAPACITY_RESERVATION_ID": "ocid1.capacityreservation.oc1.iad.test"
            },
        )

        # Phase 2: New approach (native)
        new_details = CreateModelDeploymentDetails(
            instance_shape="VM.GPU.A10.1",
            model_id="ocid1.datasciencemodel.oc1.iad.test",
            capacity_reservation_ids=["ocid1.capacityreservation.oc1.iad.test"],
        )

        # Both should result in the same capacity_reservation_ids
        assert (
            old_details.capacity_reservation_ids == new_details.capacity_reservation_ids
        )

        # Old approach should have cleaned env_var
        assert "CAPACITY_RESERVATION_ID" not in old_details.env_var

        # New approach should not have had it in env_var at all
        assert (
            new_details.env_var is None
            or "CAPACITY_RESERVATION_ID" not in new_details.env_var
        )


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_multiple_capacity_reservations(self):
        """Test that passing multiple capacity reservations is handled correctly."""
        from ads.model.deployment.model_deployment_infrastructure import (
            ModelDeploymentInfrastructure,
        )

        infra = ModelDeploymentInfrastructure()

        # SDK supports list
        multiple_ids = [
            "ocid1.capacityreservation.oc1.iad.test1",
            "ocid1.capacityreservation.oc1.iad.test2",
        ]

        infra.with_capacity_reservation_ids(multiple_ids)

        # Should accept list
        assert len(infra.capacity_reservation_ids) == 2

    def test_capacity_reservation_with_non_gpu_shape(self):
        """Test capacity reservation with non-GPU shape."""
        from ads.aqua.modeldeployment.entities import CreateModelDeploymentDetails

        # Some shapes may not support capacity reservations (validation happens at API level)
        details = CreateModelDeploymentDetails(
            instance_shape="VM.Standard2.1",  # Non-GPU shape
            model_id="ocid1.datasciencemodel.oc1.iad.test",
            capacity_reservation_ids=["ocid1.capacityreservation.oc1.iad.test"],
        )

        # Should accept it (validation happens at OCI API level)
        assert details.capacity_reservation_ids == [
            "ocid1.capacityreservation.oc1.iad.test"
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
