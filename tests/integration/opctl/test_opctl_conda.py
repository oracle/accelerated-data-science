#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os, stat
from ads.opctl.conda.cmds import create, install, publish
from ads.opctl.conda.cli import create as cli_create
from ads.opctl.conda.cli import install as cli_install
from ads.opctl.conda.cli import publish as cli_publish
from tests.integration.config import secrets
import shutil

import tempfile
import yaml

from click.testing import CliRunner

ADS_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# When running in TeamCity we specify dir, which is CHECKOUT_DIR="%teamcity.build.checkoutDir%" and
# it has permissions to remove (rmtree) folder in it, which is done by install command below
WORK_DIR = os.getenv("CHECKOUT_DIR", None)


class TestCondaRun:
    def test_conda_install_service_pack_path(self):
        runner = CliRunner()
        res = runner.invoke(
            cli_install,
            args=[
                "-s",
                "dataexpl_p37_cpu_v2",
                "-p",
                "oci://service_conda_packs@ociodscdev/service_pack",
                "--ads-config",
                ADS_CONFIG_DIR,
            ],
        )
        assert res.exit_code == 0, res.output

    def test_conda_create_publish_setup(self):
        with tempfile.TemporaryDirectory(dir=WORK_DIR) as td:
            # TeamCity fails to remove TemporaryDirectory, we adjust permissions here:
            os.chmod(td, 0o700)
            with open(os.path.join(td, "env.yaml"), "w") as f:
                f.write(
                    """
channels:
  - defaults
  - conda-forge
dependencies:
  - click
  - pip:
    - tqdm
                """
                )
            create(
                name="test Abc",
                version="1",
                environment_file=os.path.join(td, "env.yaml"),
                conda_pack_folder=os.path.join(td, "conda"),
            )
            assert os.path.exists(
                os.path.join(td, "conda", "testabc_v1", "testabc_v1_manifest.yaml")
            )

            publish(
                slug="testabc_v1",
                overwrite=True,
                conda_pack_folder=os.path.join(td, "conda"),
                ads_config=ADS_CONFIG_DIR,
            )

            def del_rw(action, name, exc):
                os.chmod(name, stat.S_IWRITE)
                os.remove(name)

            # TeamCity fails to remove files in folder, we force to remove them:
            shutil.rmtree(os.path.join(td, "conda"), onerror=del_rw)

            install(
                slug="testabc_v1",
                conda_pack_folder=os.path.join(td, "conda"),
                ads_config=ADS_CONFIG_DIR,
                overwrite=True,
            )

            assert os.path.exists(
                os.path.join(td, "conda", "testabc_v1", "testabc_v1_manifest.yaml")
            )

            with open(
                os.path.join(td, "conda", "testabc_v1", "testabc_v1_manifest.yaml")
            ) as f:
                env = yaml.safe_load(f.read())
            assert "manifest" in env
            manifest = env["manifest"]
            assert manifest["type"] == "published"
            assert manifest["slug"] == "testabc_v1"
            assert manifest["name"] == "test Abc"

    def test_conda_cli(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory(dir=WORK_DIR) as td:
            # TeamCity fails to remove TemporaryDirectory, we adjust permissions here:
            os.chmod(td, 0o700)
            with open(os.path.join(td, "env.yaml"), "w") as f:
                f.write(
                    """
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.7
  - click
                """
                )
            os.makedirs(os.path.join(td, "conda", "test1_v1"))
            res = runner.invoke(
                cli_create,
                args=[
                    "-n",
                    "test1",
                    "-f",
                    os.path.join(td, "env.yaml"),
                    "--conda-pack-folder",
                    os.path.join(td, "conda"),
                    "-o",
                ],
            )
            assert res.exit_code == 0, res.output

            res = runner.invoke(
                cli_publish,
                args=[
                    "-s",
                    "test1_v1",
                    "-o",
                    "--conda-pack-folder",
                    os.path.join(td, "conda"),
                    "--ads-config",
                    ADS_CONFIG_DIR,
                ],
            )
            assert res.exit_code == 0, res.output

            res = runner.invoke(
                cli_publish,
                args=[
                    "-s",
                    "test1_v1",
                    "-p",
                    f"oci://{secrets.opctl.BUCKET}@{secrets.common.NAMESPACE}/test-int-opctl-conda/",
                    "--conda-pack-folder",
                    os.path.join(td, "conda"),
                    "--ads-config",
                    ADS_CONFIG_DIR,
                ],
                input="test2\no\n",
            )
            assert res.exit_code == 0, res.output

            res = runner.invoke(
                cli_install,
                args=[
                    "-s",
                    "test1_v1",
                    "--conda-pack-folder",
                    os.path.join(td, "conda"),
                    "--ads-config",
                    ADS_CONFIG_DIR,
                ],
                input="o\n",
            )
            assert res.exit_code == 0, res.output

            res = runner.invoke(
                cli_install,
                args=[
                    "-u",
                    f"oci://{secrets.opctl.BUCKET}@{secrets.common.NAMESPACE}/test-int-opctl-conda/test2",
                    "--conda-pack-folder",
                    os.path.join(td, "conda"),
                    "--ads-config",
                    ADS_CONFIG_DIR,
                ],
            )
            assert res.exit_code == 0, res.output
            assert os.path.exists(os.path.join(td, "conda", "test2"))
