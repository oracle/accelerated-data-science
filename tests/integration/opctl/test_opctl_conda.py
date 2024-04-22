#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from ads.common.auth import AuthType
from ads.opctl.conda.cmds import create, install, publish
from ads.opctl.conda.cli import create as cli_create
from ads.opctl.conda.cli import install as cli_install
from ads.opctl.conda.cli import publish as cli_publish
from tests.integration.config import secrets

import tempfile
import yaml

from click.testing import CliRunner

ADS_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

if "TEAMCITY_VERSION" in os.environ:
    # When running in TeamCity we specify dir, which is CHECKOUT_DIR="%teamcity.build.checkoutDir%"
    WORK_DIR = os.getenv("CHECKOUT_DIR", "~")
    CONDA_PACK_FOLDER = f"{WORK_DIR}/conda"
    AUTH = AuthType.INSTANCE_PRINCIPAL
else:
    CONDA_PACK_FOLDER = "~/conda"
    AUTH = AuthType.SECURITY_TOKEN


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
                "--conda-pack-folder",
                CONDA_PACK_FOLDER,
                "--auth",
                AUTH,
            ],
        )
        assert res.exit_code == 0, res.output

    def test_conda_create_publish_setup(self):
        td = tempfile.TemporaryDirectory(dir=WORK_DIR)
        td_name = td.name
        with open(os.path.join(td_name, "env.yaml"), "w") as f:
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
            environment_file=os.path.join(td_name, "env.yaml"),
            conda_pack_folder=os.path.join(td_name, "conda"),
        )
        assert os.path.exists(
            os.path.join(td_name, "conda", "testabc_v1", "testabc_v1_manifest.yaml")
        )

        publish(
            slug="testabc_v1",
            overwrite=True,
            conda_pack_folder=os.path.join(td_name, "conda"),
            ads_config=ADS_CONFIG_DIR,
            auth=AUTH,
        )

        td = tempfile.TemporaryDirectory(dir=WORK_DIR)
        td_name = td.name
        install(
            slug="testabc_v1",
            conda_pack_folder=os.path.join(td_name, "conda"),
            ads_config=ADS_CONFIG_DIR,
            overwrite=True,
            auth=AUTH,
        )

        assert os.path.exists(
            os.path.join(td_name, "conda", "testabc_v1", "testabc_v1_manifest.yaml")
        )

        with open(
            os.path.join(td_name, "conda", "testabc_v1", "testabc_v1_manifest.yaml")
        ) as f:
            env = yaml.safe_load(f.read())
        assert "manifest" in env
        manifest = env["manifest"]
        assert manifest["type"] == "published"
        assert manifest["slug"] == "testabc_v1"
        assert manifest["name"] == "test Abc"

    def test_conda_cli(self):
        runner = CliRunner()
        td = tempfile.TemporaryDirectory(dir=WORK_DIR)
        td_name = td.name
        with open(os.path.join(td_name, "env.yaml"), "w") as f:
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
        os.makedirs(os.path.join(td_name, "conda", "test1_v1"))
        res = runner.invoke(
            cli_create,
            args=[
                "-n",
                "test1",
                "-f",
                os.path.join(td_name, "env.yaml"),
                "--conda-pack-folder",
                os.path.join(td_name, "conda"),
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
                os.path.join(td_name, "conda"),
                "--ads-config",
                ADS_CONFIG_DIR,
                "--auth",
                AUTH,
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
                os.path.join(td_name, "conda"),
                "--ads-config",
                ADS_CONFIG_DIR,
                "--auth",
                AUTH,
            ],
            input="test2\no\n",
        )
        assert res.exit_code == 0, res.output

        res = runner.invoke(
            cli_install,
            args=[
                "-u",
                f"oci://{secrets.opctl.BUCKET}@{secrets.common.NAMESPACE}/test-int-opctl-conda/test2",
                "--conda-pack-folder",
                os.path.join(td_name, "conda"),
                "--ads-config",
                ADS_CONFIG_DIR,
                "--auth",
                AUTH,
            ],
        )
        assert res.exit_code == 0, res.output
        assert os.path.exists(os.path.join(td_name, "conda", "test2"))
