#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os

from click.testing import CliRunner

from ads.common.auth import AuthType
from ads.jobs.cli import run, watch, delete


class TestJobsCLI:
    # TeamCity will use Instance Principal, when running locally - set OCI_IAM_TYPE to security_token
    auth = os.environ.get("OCI_IAM_TYPE", AuthType.INSTANCE_PRINCIPAL)

    def test_create_watch_delete_job(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        runner = CliRunner()
        res = runner.invoke(
            run,
            args=[
                "-f",
                os.path.join(curr_dir, "../yamls", "sample_job.yaml"),
                "--auth",
                self.auth,
            ],
        )
        assert res.exit_code == 0, res.output
        run_id = res.output.split("\n")[1]
        res2 = runner.invoke(
            watch,
            args=[
                run_id,
                "--auth",
                self.auth,
            ],
        )
        assert res2.exit_code == 0, res2.output

        res3 = runner.invoke(
            delete,
            args=[
                run_id,
                "--auth",
                self.auth,
            ],
        )
        assert res3.exit_code == 0, res3.output

    def test_create_watch_delete_dataflow(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        runner = CliRunner()
        res = runner.invoke(
            run,
            args=[
                "-f",
                os.path.join(curr_dir, "../yamls", "sample_dataflow.yaml"),
                "--auth",
                self.auth,
            ],
        )
        assert res.exit_code == 0, res.output
        run_id = res.output.split("\n")[1]
        res2 = runner.invoke(
            watch,
            args=[
                run_id,
                "--auth",
                self.auth,
            ],
        )
        assert res2.exit_code == 0, res2.output

        res3 = runner.invoke(
            run, args=["-f", os.path.join(curr_dir, "../yamls", "sample_dataflow.yaml")]
        )
        run_id2 = res3.output.split("\n")[1]
        res4 = runner.invoke(
            delete,
            args=[
                run_id2,
                "--auth",
                self.auth,
            ],
        )
        assert res4.exit_code == 0, res4.output
