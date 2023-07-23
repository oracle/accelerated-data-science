#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import pytest

from ads.opctl.config.utils import read_from_ini

try:
    from ads.opctl.cmds import cancel, configure, delete, init, watch
except ImportError:
    raise unittest.SkipTest("ADS OPCTL is not available. Skipping the tests.")


class TestConfigureCmd:
    @patch("ads.opctl.cmds.click.prompt")
    @patch("ads.opctl.cmds.click.confirm")
    def test_configure(self, confirm, prompt, monkeypatch):
        monkeypatch.delenv("NB_SESSION_OCID", raising=False)
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "oci_config"), mode="w") as f:
                f.write(
                    """
[DEFAULT]
user = xxxxx
fingerprint = xxxxxx
tenancy = xxxxx
region = us-ashburn-1
key_file = ~/.oci/oci_api_key.pem
                """
                )
            prompt.side_effect = (
                [
                    os.path.join(td, "ads_ops"),
                    os.path.join(td, "oci_config"),
                    "DEFAULT",
                    ".",
                    "oci://bucket@namespace/path",
                ]
                + ["abc"] * 8
                + ["oci://bucket@namespace/path"]
                + ["abc"] * 2
                + ["abc"] * 4
                + ["oci://bucket@namespace/path"]
                + ["abc"] * 7
                + ["abc"] * 4
                + ["abc"] * 10
                + ["1"]
                + ["3"]
            )
            confirm.return_value = "y"
            configure()
            parser = read_from_ini(os.path.join(td, "ads_ops", "config.ini"))
            assert parser["OCI"]["oci_config"] == os.path.join(td, "oci_config")
            assert parser["OCI"]["oci_profile"] == "DEFAULT"
            assert (
                os.path.abspath(parser["CONDA"]["conda_pack_folder"])
                == parser["CONDA"]["conda_pack_folder"]
            )
            assert os.path.exists(os.path.join(td, "ads_ops", "ml_job_config.ini"))
            parser = read_from_ini(os.path.join(td, "ads_ops", "ml_job_config.ini"))
            assert (
                parser["DEFAULT"]["conda_pack_os_prefix"]
                == "oci://bucket@namespace/path"
            )
            assert os.path.exists(os.path.join(td, "ads_ops", "dataflow_config.ini"))
            parser = read_from_ini(os.path.join(td, "ads_ops", "dataflow_config.ini"))
            assert parser["DEFAULT"]["script_bucket"] == "oci://bucket@namespace/path"
            assert os.path.exists(os.path.join(td, "ads_ops", "local_backend.ini"))
            parser = read_from_ini(os.path.join(td, "ads_ops", "local_backend.ini"))
            assert parser["DEFAULT"]["max_parallel_containers"] == "1"
            assert parser["DEFAULT"]["pipeline_status_poll_interval_seconds"] == "3"
            with pytest.raises(ValueError):
                prompt.side_effect = [
                    os.path.join(td, "ads_ops"),
                    "~/.oci/config",
                    "NONEXIST",
                    ".",
                ]
                configure()

    @patch("ads.opctl.cmds.click.prompt")
    @patch("ads.opctl.cmds.click.confirm")
    def test_configure_in_notebook_session(self, confirm, prompt, monkeypatch):
        monkeypatch.setenv("NB_SESSION_OCID", "abced")
        nbsession = Mock()
        nbsession.compartment_id = "compartment_id"
        nbsession.project_id = ("project_id",)
        notebook_session_configuration_details = Mock()
        notebook_session_configuration_details.subnet_id = "subnet_id"
        notebook_session_configuration_details.block_storage_size_in_gbs = (
            "block_storage_size_in_gbs"
        )
        notebook_session_configuration_details.shape = "shape"
        nbsession.notebook_session_configuration_details = (
            notebook_session_configuration_details
        )

        with patch(
            "ads.opctl.cmds.DSCNotebookSession.from_ocid", return_value=nbsession
        ):
            with tempfile.TemporaryDirectory() as td:
                prompt.side_effect = (
                    [
                        os.path.join(td, "ads_ops"),
                        os.path.join(td, "oci_config"),
                        "DEFAULT",
                        ".",
                        "oci://bucket@namespace/path",
                    ]
                    + ["abc"] * 8
                    + ["oci://bucket@namespace/path"]
                    + ["abc"] * 2
                    + ["abc"] * 4
                    + ["oci://bucket@namespace/path"]
                    + ["abc"] * 7
                    + ["abc"] * 4
                    + ["abc"] * 10
                    + ["1"]
                    + ["3"]
                )
                confirm.return_value = "y"
                configure()
                parser = read_from_ini(os.path.join(td, "ads_ops", "config.ini"))
                assert parser["OCI"]["oci_config"] == os.path.join(td, "oci_config")
                assert parser["OCI"]["oci_profile"] == "DEFAULT"
                assert (
                    os.path.abspath(parser["CONDA"]["conda_pack_folder"])
                    == parser["CONDA"]["conda_pack_folder"]
                )
                assert os.path.exists(os.path.join(td, "ads_ops", "ml_job_config.ini"))
                parser = read_from_ini(os.path.join(td, "ads_ops", "ml_job_config.ini"))
                assert parser["RESOURCE_PRINCIPAL"]["compartment_id"] == "abc"
                assert (
                    parser["RESOURCE_PRINCIPAL"]["conda_pack_os_prefix"]
                    == "oci://bucket@namespace/path"
                )
                assert os.path.exists(
                    os.path.join(td, "ads_ops", "dataflow_config.ini")
                )
                parser = read_from_ini(
                    os.path.join(td, "ads_ops", "dataflow_config.ini")
                )
                assert (
                    parser["RESOURCE_PRINCIPAL"]["script_bucket"]
                    == "oci://bucket@namespace/path"
                )
                assert os.path.exists(os.path.join(td, "ads_ops", "local_backend.ini"))
                parser = read_from_ini(os.path.join(td, "ads_ops", "local_backend.ini"))
                assert parser["RESOURCE_PRINCIPAL"]["max_parallel_containers"] == "1"
                assert (
                    parser["RESOURCE_PRINCIPAL"][
                        "pipeline_status_poll_interval_seconds"
                    ]
                    == "3"
                )

    @patch("ads.opctl.backend.ads_ml_pipeline.PipelineBackend.watch")
    @patch("ads.opctl.backend.ads_ml_job.MLJobBackend.watch")
    def test_watch(self, job_watch_func, pipeline_watch_func, monkeypatch):
        monkeypatch.delenv("NB_SESSION_OCID", raising=False)
        watch(ocid="...datasciencejobrun...")
        job_watch_func.assert_called()
        with pytest.raises(ValueError):
            watch(ocid="....datasciencejob....")

        watch(ocid="...datasciencepipelinerun...")
        pipeline_watch_func.assert_called()
        with pytest.raises(ValueError):
            watch(ocid="....datasciencepipeline....")

    @patch("ads.opctl.backend.ads_ml_pipeline.PipelineBackend.cancel")
    @patch("ads.opctl.backend.ads_ml_job.MLJobBackend.cancel")
    def test_cancel(self, job_cancel_func, pipeline_cancel_func, monkeypatch):
        monkeypatch.delenv("NB_SESSION_OCID", raising=False)
        cancel(ocid="...datasciencejobrun...")
        job_cancel_func.assert_called()
        with pytest.raises(ValueError):
            cancel(ocid="....datasciencejob....")

        cancel(ocid="...datasciencepipelinerun...")
        pipeline_cancel_func.assert_called()
        with pytest.raises(ValueError):
            cancel(ocid="....datasciencepipeline....")

    @patch("ads.opctl.backend.ads_ml_job.MLJobBackend.delete")
    def test_delete(self, delete_func, monkeypatch):
        monkeypatch.delenv("NB_SESSION_OCID", raising=False)
        delete(ocid="...datasciencejobrun...")
        delete_func.assert_called()
        delete(ocid="....datasciencejob....")
        delete_func.assert_called()

    @patch("ads.opctl.backend.ads_ml_job.MLJobBackend.init")
    def test_init_success(self, init_func, monkeypatch):
        """Tests generating a starter specification template YAML for the Data Science resource."""
        monkeypatch.delenv("NB_SESSION_OCID", raising=False)
        init(
            resource_type="job",
            runtime_type="container",
            output="test.yaml",
            overwrite=True,
        )
        init_func.assert_called_with(
            uri="test.yaml", overwrite=True, runtime_type="container"
        )

    def test_init_fail(self, monkeypatch):
        """Ensures that generating a starter YAML specification fails in case of wrong input params."""
        monkeypatch.delenv("NB_SESSION_OCID", raising=False)
        with pytest.raises(ValueError):
            init(resource_type=None)
