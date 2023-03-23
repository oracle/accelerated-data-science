#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest
from mock import patch, MagicMock
import tempfile
import os
from pathlib import Path

import yaml

from ads.common.auth import AuthType
from ads.opctl.conda.cmds import create, install, publish
from ads.opctl.conda.multipart_uploader import MultiPartUploader
from ads.opctl.constants import ML_JOB_IMAGE


class TestOpctlConda:
    @patch("ads.opctl.conda.cmds.run_command")
    @patch("ads.opctl.conda.cmds.get_docker_client")
    @patch("ads.opctl.conda.cmds.run_container")
    def test_conda_create(
        self, mock_run_container, mock_docker, mock_run_cmd, monkeypatch
    ):
        with pytest.raises(FileNotFoundError):
            create(slug="test Abc", environment_file="environment.yaml")
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "env.yaml"), "w") as f:
                f.write(
                    """
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.7
  - click
  - numpy
                """
                )
            create(
                name="test",
                version=str(1),
                environment_file=os.path.join(td, "env.yaml"),
                conda_pack_folder=os.path.join(td, "conda"),
                gpu=True,
            )
            mock_run_container.assert_called_with(
                image="ml-job-gpu",
                bind_volumes={
                    os.path.join(td, "conda", "test_v1"): {
                        "bind": "/home/datascience/test_v1"
                    },
                    os.path.join(td, "env.yaml"): {
                        "bind": "/home/datascience/env.yaml"
                    },
                },
                env_vars={},
                command="conda env create --prefix /home/datascience/test_v1 --file /home/datascience/env.yaml",
            )
            assert os.path.exists(
                os.path.join(td, "conda", "test_v1", "test_v1_manifest.yaml")
            )
            with open(
                os.path.join(td, "conda", "test_v1", "test_v1_manifest.yaml")
            ) as f:
                manifest = yaml.safe_load(f.read())

            assert manifest["channels"] == ["defaults", "conda-forge"]
            assert manifest["dependencies"] == ["python=3.7", "click", "numpy"]
            assert manifest["manifest"]["name"] == "test"
            assert manifest["manifest"]["slug"] == "test_v1"
            assert manifest["manifest"]["type"] == "published"
            assert manifest["manifest"]["arch_type"] == "GPU"

            monkeypatch.setenv("NB_SESSION_OCID", "abcde")
            create(
                name="test2",
                version=str(1),
                environment_file=os.path.join(td, "env.yaml"),
                conda_pack_folder=os.path.join(td, "conda"),
                gpu=True,
            )
            mock_run_cmd.assert_called_with(
                f"conda env create --prefix {os.path.join(td, 'conda', 'test2_v1')} --file {os.path.join(td, 'env.yaml')}",
                shell=True,
            )

    @patch("ads.opctl.conda.cmds.run_command")
    @patch("ads.opctl.conda.cmds.get_docker_client")
    @patch("os.path.getsize", return_value=200 * 1024 * 1024)
    @patch("ads.opctl.conda.cmds.MultiPartUploader", autospec=True)
    @patch("ads.opctl.conda.cmds.run_container")
    def test_conda_publish(
        self,
        mock_run_container,
        mock_uploader,
        mock_getsize,
        mock_docker,
        mock_run_cmd,
        monkeypatch,
    ):

        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(FileNotFoundError):
                publish(
                    slug="test",
                    overwrite=True,
                    conda_pack_folder=os.path.join(td, "conda"),
                    conda_pack_os_prefix="oci://bucket@ns/path",
                )
            os.makedirs(os.path.join(td, "conda", "test"))
            with open(
                os.path.join(td, "conda", "test", "test_manifest.yaml"), "w"
            ) as f:
                f.write(
                    """
channels:
  - defaults
dependencies:
  - python
  - click
manifest:
  name: test
  slug: test
  version: 1.0
  type: published
  arch_type: GPU
                """
                )
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            Path(os.path.join(td, "conda", "test", "test.tar.gz")).touch()
            publish(
                slug="test",
                overwrite=True,
                conda_pack_os_prefix="oci://bucket@ns/path",
                conda_pack_folder=os.path.join(td, "conda"),
                oci_config="~/.oci/config",
                oci_profile="DEFAULT",
            )

            mock_run_container.assert_called_with(
                image="ml-job-gpu",
                bind_volumes={
                    os.path.join(td, "conda", "test"): {
                        "bind": "/home/datascience/test"
                    },
                    os.path.normpath(
                        os.path.join(
                            curr_dir,
                            "..",
                            "..",
                            "..",
                            "..",
                            "ads",
                            "opctl",
                            "conda",
                            "pack.py",
                        )
                    ): {"bind": "/home/datascience/pack.py"},
                },
                env_vars={},
                command="python /home/datascience/pack.py /home/datascience/test",
            )

            mock_uploader.assert_called_with(
                os.path.join(td, "conda", "test", "test.tar.gz"),
                "oci://bucket@ns/path/gpu/test/1.0/test",
                10,
                "~/.oci/config",
                "DEFAULT",
                AuthType.API_KEY,
            )

            monkeypatch.setenv("NB_SESSION_OCID", "abced")
            Path(os.path.join(td, "conda", "test", "test.tar.gz")).touch()
            publish(
                slug="test",
                overwrite=True,
                conda_pack_os_prefix="oci://bucket@ns/path",
                conda_pack_folder=os.path.join(td, "conda"),
                oci_config="~/.oci/config",
                oci_profile="DEFAULT",
                auth=AuthType.API_KEY,
            )
            import ads

            mock_run_cmd.assert_called_with(
                f"python {os.path.join(ads.__path__[0], 'opctl', 'conda', 'pack.py')} {os.path.join(td, 'conda', 'test')}",
                shell=True,
            )

    @patch("ads.opctl.conda.cmds.run_command")
    @patch("ads.opctl.conda.cmds.get_docker_client")
    @patch("ads.opctl.conda.cmds.run_container")
    @patch("ads.opctl.conda.cmds.subprocess.Popen")
    @patch("ads.opctl.conda.cmds.click.prompt", return_value="q")
    def test_install_conda(
        self, prompt, popen, mock_run_container, mock_docker, mock_run_cmd, monkeypatch
    ):
        process = MagicMock()
        process.returncode = 0
        popen.return_value = process
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "slug"))
            with open(os.path.join(td, "slug", "_manifest.yaml"), "w") as f:
                f.write(
                    """
manifest:
    pack_path: xxxx
                        """
                )
            Path(os.path.join(td, "slug.tar.gz")).touch()
            os.makedirs(os.path.join(td, "slug", "bin"))
            Path(os.path.join(td, "slug", "bin", "conda-unpack")).touch()
            install(
                conda_uri="oci://bucket@namespace/path/slug",
                conda_pack_folder=td,
                oci_config=None,
                oci_profile=None,
                overwrite=True,
            )
            assert popen.call_count == 2
            mock_run_container.assert_called_with(
                image=ML_JOB_IMAGE,
                bind_volumes={
                    os.path.join(td, "slug"): {"bind": "/home/datascience/slug"}
                },
                env_vars={},
                command="/home/datascience/slug/bin/conda-unpack",
            )

            monkeypatch.setenv("NB_SESSION_OCID", "abced")
            Path(os.path.join(td, "slug.tar.gz")).touch()
            install(
                conda_uri="oci://bucket@namespace/path/slug",
                conda_pack_folder=td,
                oci_config=None,
                oci_profile=None,
                overwrite=True,
            )
            mock_run_cmd.assert_called_with(
                os.path.join(td, "slug", "bin", "conda-unpack")
            )


class TestMultiPartUploader:
    @patch("ads.opctl.conda.multipart_uploader.create_signer")
    @patch("ads.opctl.conda.multipart_uploader.mmap")
    @patch("ads.opctl.conda.multipart_uploader.OCIClientFactory.object_storage")
    @patch.object(MultiPartUploader, "_chunks")
    @patch.object(MultiPartUploader, "_upload_chunk")
    @patch.object(MultiPartUploader, "_track_progress")
    def test_upload_multipart(
        self, track_progress, upload_chunk, chunks, oci_client, mmap, signer, tmp_path
    ):
        oci_client.create_multipart_upload = MagicMock()
        oci_client.upload_part = MagicMock()
        oci_client.commit_multipart_upload = MagicMock()

        mmap.mmap = MagicMock()
        path = tmp_path / "pack_file"
        path.touch()
        uploader = MultiPartUploader(
            path,
            "oci://bucket@namespace/path",
            10,
            oci_profile="DEFAULT",
            auth_type=AuthType.API_KEY,
        )
        assert uploader.upload()
