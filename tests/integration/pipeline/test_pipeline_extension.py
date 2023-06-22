#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import unittest
import pytest
import builtins

try:
    from ads.pipeline import Pipeline, PipelineRun
except (ImportError, AttributeError) as e:
    raise unittest.SkipTest(
        "OCI MLPipeline is not available. Skipping the MLPipeline tests."
    )
from IPython.testing.globalipapp import start_ipython
from IPython.utils.io import capture_output


class TestPipelineExt:
    @pytest.fixture(scope="class")
    def ip(self):
        start_ipython()
        ip = builtins.ip
        ip.run_line_magic("load_ext", "ads.pipeline.extension")
        yield ip

    def test_pipeline_help(self, ip):
        with capture_output() as captured:
            ip.run_line_magic("pipeline", "-h")
        stdout = captured.stdout
        assert "Usage: pipeline [SUBCOMMAND]" in stdout
        assert "Run pipeline [SUBCOMMAND] -h to see more details." in stdout
        with capture_output() as captured:
            ip.run_line_magic("pipeline", "run -h")
        stdout = captured.stdout
        assert "Usage: pipeline run [OPTIONS]" in stdout
        with capture_output() as captured:
            ip.run_line_magic("pipeline", "log -h")
        stdout = captured.stdout
        assert "Usage: pipeline log [OPTIONS] [RUN_ID]" in stdout
        with capture_output() as captured:
            ip.run_line_magic("pipeline", "cancel -h")
        stdout = captured.stdout
        assert "Usage: pipeline cancel [RUN_ID]" in stdout
        with capture_output() as captured:
            ip.run_line_magic("pipeline", "delete -h")
        stdout = captured.stdout
        assert "Usage: pipeline delete [OCID]" in stdout
        with capture_output() as captured:
            ip.run_line_magic("pipeline", "show -h")
        stdout = captured.stdout
        assert "Usage: pipeline show [OCID]" in stdout
        with capture_output() as captured:
            ip.run_line_magic("pipeline", "status -h")
        stdout = captured.stdout
        assert "Usage: pipeline status [OPTIONS] [RUN_ID]" in stdout

    def test_pipeline_run(self, ip):
        with capture_output() as captured:
            yaml_path = self.get_yaml_path()
            ip.run_line_magic("pipeline", f"run --file {yaml_path}")
        stdout = captured.stdout
        assert "Pipeline OCID:" in stdout, stdout
        assert "Pipeline Run OCID:" in stdout, stdout

    def test_pipeline_magic_commands(self, ip):
        yaml_path = self.get_yaml_path()
        pipeline = Pipeline.from_yaml(uri=yaml_path)
        pipeline.create()
        pipeline_run = pipeline.run()

        with capture_output() as captured:
            ip.run_line_magic("pipeline", f"show {pipeline.id}")
        stdout = captured.stdout
        assert len(stdout.strip("\n").split("\n")) != 0, stdout

        with capture_output() as captured:
            ip.run_line_magic("pipeline", f"cancel {pipeline_run.id}")
        stdout = captured.stdout
        assert f"Pipeline Run {pipeline_run.id} has been cancelled." in stdout

        assert (
            pipeline_run.sync().lifecycle_state == PipelineRun.LIFECYCLE_STATE_CANCELED
        )

        with capture_output() as captured:
            ip.run_line_magic("pipeline", f"log {pipeline_run.id}")
        stdout = captured.stdout
        assert len(stdout.strip("\n").split("\n")) != 0, stdout

        with capture_output() as captured:
            ip.run_line_magic("pipeline", f"status {pipeline_run.id}")
        stdout = captured.stdout
        assert len(stdout.strip("\n").split("\n")) != 0, stdout

        with capture_output() as captured:
            ip.run_line_magic("pipeline", f"delete {pipeline_run.id}")
        stdout = captured.stdout
        assert f"Pipeline Run {pipeline_run.id} has been deleted." in stdout

        assert (
            pipeline_run.sync().lifecycle_state == PipelineRun.LIFECYCLE_STATE_DELETED
        )

        with capture_output() as captured:
            ip.run_line_magic("pipeline", f"delete {pipeline.id}")
        stdout = captured.stdout
        assert f"Pipeline {pipeline.id} has been deleted." in stdout

        assert (
            pipeline.data_science_pipeline.sync().lifecycle_state
            == Pipeline.LIFECYCLE_STATE_DELETED
        )

    def get_yaml_path(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(curr_dir, "..", "yamls", "sample_pipeline.yaml")
        return yaml_path
