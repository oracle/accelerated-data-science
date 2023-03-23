#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import tempfile
import os
from unittest.mock import patch

from ads.opctl.backend.ads_ml_job import MLJobBackend
from ads.jobs import Job, DataScienceJobRun


class TestMLJobBackend:
    @property
    def curr_dir(self):
        return os.path.dirname(os.path.abspath(__file__))

    @property
    def config(self):
        return {
            "execution": {
                "backend": "job",
                "use_conda": True,
                "debug": False,
                "env_var": ["TEST_ENV=test_env"],
                "auth": "api_key",
                "oci_config": "~/.oci/config",
                "oci_profile": "DEFAULT",
                "command": "-n hello-world -c ~/.oci/config -p DEFAULT",
                "env_vars": {"TEST_ENV": "test_env"},
                "job_name": "hello-world",
            },
            "infrastructure": {
                "compartment_id": "ocid1.compartment.oc1..<unique_id>",
                "project_id": "ocid1.datascienceproject.oc1.<unique_id>",
                "subnet_id": "ocid1.subnet.oc1.iad.<unique_id>",
                "log_group_id": "ocid1.loggroup.oc1.iad.<unique_id>",
                "log_id": "ocid1.log.oc1.iad.<unique_id>",
                "shape_name": "VM.Standard2.1",
                "block_storage_size": 50,
            },
        }

    @patch(
        "ads.opctl.backend.ads_ml_job.Job.run",
        return_value=DataScienceJobRun(id="TestRunId"),
    )
    @patch(
        "ads.opctl.backend.ads_ml_job.Job.create",
        return_value=Job(name="TestJob"),
    )
    @patch("ads.opctl.backend.ads_ml_job.Job.from_dict", return_value=Job())
    def test_apply(self, mock_from_dict, mock_create, mock_run):
        config = self.config
        backend = MLJobBackend(config)
        backend.apply()
        mock_from_dict.assert_called_with(config)
        mock_create.assert_called_with()
        mock_run.assert_called_with()

    def test_create_payload(self):
        payload = MLJobBackend(self.config)._create_payload()
        assert payload.infrastructure.block_storage_size == 50
        assert payload.infrastructure.shape_name == "VM.Standard2.1"
        assert payload.name == "hello-world"
        assert payload.infrastructure.type == "dataScienceJob"

    @patch("ads.opctl.backend.ads_ml_job.Job.create", return_value=Job())
    @patch("ads.opctl.backend.ads_ml_job.Job.run")
    @patch("ads.opctl.backend.ads_ml_job.Job.runtime")
    def test_run_with_conda_pack(self, rt, job_run, job_create):
        config = self.config
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "src"))
            config["execution"]["conda_type"] = "published"
            config["execution"]["entrypoint"] = "main.py"
            config["execution"]["conda_uri"] = "oci://bucket@namespace/path/slug"
            payload = MLJobBackend(config)._create_payload()
            MLJobBackend(config)._run_with_conda_pack(payload, os.path.join(td, "src"))
            rt.with_custom_conda.assert_called_with("oci://bucket@namespace/path/slug")
            rt.with_source.assert_called()
            rt.set_spec.assert_called_with(
                "args", ["-n", "hello-world", "-c", "~/.oci/config", "-p", "DEFAULT"]
            )
            job_create.assert_called()
            job_run.assert_called()

    @patch("ads.opctl.backend.ads_ml_job.Job.create", return_value=Job())
    @patch("ads.opctl.backend.ads_ml_job.Job.run")
    @patch("ads.opctl.backend.ads_ml_job.Job.runtime")
    def test_run_with_image(self, rt, job_run, job_create):
        config = self.config
        config["execution"]["image"] = "docker.io/image"
        config["execution"]["operator_name"] = "hello-world"
        MLJobBackend(config)._run_with_image(Job())
        rt.with_image.assert_called_with("docker.io/image:latest")
        rt.with_cmd.assert_called_with(
            "python,/etc/datascience/operators/run.py,-r,-n,hello-world,-c,~/.oci/config,-p,DEFAULT"
        )

        config["execution"].pop("operator_name")
        config["execution"]["entrypoint"] = "python main.py"
        MLJobBackend(config)._run_with_image(Job())
        rt.with_entrypoint.assert_called_with("python main.py")
        rt.with_cmd.assert_called_with("-n,hello-world,-c,~/.oci/config,-p,DEFAULT")
        job_create.assert_called()
        job_run.assert_called()
