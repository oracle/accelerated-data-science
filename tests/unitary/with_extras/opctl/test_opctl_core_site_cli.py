#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from shlex import split
import os
import tempfile

from click.testing import CliRunner
import pytest
from pathlib import Path

from ads.common.auth import AuthType
from ads.opctl.spark.cli import core_site


def _test_command(cmd_str):
    runner = CliRunner()
    res = runner.invoke(core_site, args=split(cmd_str))
    assert res.exit_code == 0, res.output


class TestCoreSiteCLI:
    @pytest.mark.parametrize("auth", AuthType.values())
    def test_core_site_cli(self, monkeypatch, auth):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "conda", "pyspark"))
            Path(os.path.join(td, "conda", "pyspark", "spark-defaults.conf")).touch()
            monkeypatch.setenv("CONDA_PREFIX", os.path.join(td, "conda", "pyspark"))
            _test_command(f"-a {auth} -o")
            assert os.path.exists(os.path.join(td, "conda", "pyspark", "core-site.xml"))
