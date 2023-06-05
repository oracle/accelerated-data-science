#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import unittest

from click.testing import CliRunner

try:
    from ads.pipeline.cli import delete, run, watch
except (ImportError, AttributeError) as e:
    raise unittest.SkipTest(
        "OCI MLPipeline is not available. Skipping the MLPipeline tests."
    )


class TestPipelineCLI:
    def test_create_watch_delete_pipeline(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        runner = CliRunner()
        res = runner.invoke(
            run, args=["-f", os.path.join(curr_dir, "../yamls", "sample_pipeline.yaml")]
        )
        assert res.exit_code == 0, res.output
        run_id = res.output.split("\n")[1]
        res2 = runner.invoke(watch, args=[run_id])
        assert res2.exit_code == 0, res2.output
        res3 = runner.invoke(delete, args=[run_id])
        assert res3.exit_code == 0, res3.output
