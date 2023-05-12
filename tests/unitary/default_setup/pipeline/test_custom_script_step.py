#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import unittest

try:
    from ads.pipeline.builders.infrastructure.custom_script import (
        CustomScriptStep,
    )
except ImportError:
    raise unittest.SkipTest(
        "OCI MLPipeline is not available. Skipping the MLPipeline tests."
    )


class TestCustomScriptStep:
    def test_custom_script_step_define_from_kwargs(self):
        infrastructure = CustomScriptStep(
            project_id="TestProjectId",
            compartment_id="TestCompartmentId",
            shape_name="TestShapeName",
            block_storage_size=200,
            log_id="TestLogId",
            log_group_id="TestLogGroupId",
        )

        assert infrastructure.project_id == "TestProjectId"
        assert infrastructure.compartment_id == "TestCompartmentId"
        assert infrastructure.shape_name == "TestShapeName"
        assert infrastructure.block_storage_size == 200
        assert infrastructure.log_id == "TestLogId"
        assert infrastructure.log_group_id == "TestLogGroupId"

    def test_custom_script_step_define_from_spec(self):
        infrastructure = CustomScriptStep(
            spec={
                "project_id": "TestProjectId",
                "compartment_id": "TestCompartmentId",
                "shape_name": "TestShapeName",
                "block_storage_size": 200,
                "log_id": "TestLogId",
                "log_group_id": "TestLogGroupId",
            }
        )

        assert infrastructure.project_id == "TestProjectId"
        assert infrastructure.compartment_id == "TestCompartmentId"
        assert infrastructure.shape_name == "TestShapeName"
        assert infrastructure.block_storage_size == 200
        assert infrastructure.log_id == "TestLogId"
        assert infrastructure.log_group_id == "TestLogGroupId"
