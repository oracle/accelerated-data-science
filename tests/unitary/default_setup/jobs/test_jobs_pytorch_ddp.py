#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
import unittest
import zipfile
from unittest import mock
from ads.jobs import PyTorchDistributedRuntime, DataScienceJob, DataScienceJobRun
from ads.jobs.builders.infrastructure.dsc_job_runtime import (
    PyTorchDistributedRuntimeHandler as Handler,
)
from ads.jobs.builders.runtimes.pytorch_runtime import (
    PyTorchDistributedArtifact,
    GitPythonArtifact,
)
from ads.opctl.distributed.common import cluster_config_helper as cluster
from ads.jobs.templates import driver_utils as utils


class PyTorchRuntimeHandlerTest(unittest.TestCase):
    INPUT_SRC = "oci://bucket@namespace/path/to/input"
    INPUT_DST = "data/input.txt"
    TEST_REPO = "https://github.com/pytorch/examples.git"
    TEST_COMMIT = "d91085d2181bf6342ac7dafbeee6fc0a1f64dcec"
    REPLICAS = 2
    PIP_REQ = "distributed/minGPT-ddp/requirements.txt"
    PIP_PKG = '"package>1.0"'
    TORCHRUN_CMD = (
        "torchrun distributed/minGPT-ddp/mingpt/main.py data_config.path=data/input.txt"
    )

    def init_runtime(self):
        """Initializes a PyTorchDistributedRuntime for testing."""
        return (
            PyTorchDistributedRuntime()
            .with_replica(self.REPLICAS)
            .with_service_conda("pytorch110_p38_gpu_v1")
            .with_git(
                self.TEST_REPO,
                commit=self.TEST_COMMIT,
            )
            .with_inputs({self.INPUT_SRC: self.INPUT_DST})
            .with_dependency(
                pip_req=self.PIP_REQ,
                pip_pkg=self.PIP_PKG,
            )
            .with_command(self.TORCHRUN_CMD)
        )

    def test_translate_artifact(self):
        """Tests preparing ADS driver scripts in job artifacts."""
        artifact = Handler(DataScienceJob())._translate_artifact(self.init_runtime())
        self.assertIsInstance(artifact, PyTorchDistributedArtifact)
        self.assertEqual(
            artifact.source,
            "",
            "Artifact source should be empty when using source code from Git.",
        )
        with artifact:
            self.assertTrue(
                artifact.path.endswith(
                    PyTorchDistributedArtifact.DEFAULT_BASENAME + ".zip"
                )
            )
            file_list = zipfile.ZipFile(artifact.path).namelist()
            self.assertEqual(len(file_list), 5, f"Expected 5 files. Got: {file_list}")
            self.assertIn(PyTorchDistributedArtifact.CONST_DRIVER_UTILS, file_list)
            self.assertIn(PyTorchDistributedArtifact.CONST_DRIVER_SCRIPT, file_list)
            self.assertIn(PyTorchDistributedArtifact.CONST_LIB_HOSTNAME, file_list)
            self.assertIn(PyTorchDistributedArtifact.CONST_OCI_METRICS, file_list)
            self.assertIn(GitPythonArtifact.CONST_DRIVER_SCRIPT, file_list)

    def test_translate_env(self):
        """Tests setting up environment variables"""
        envs = Handler(DataScienceJob())._translate_env(self.init_runtime())
        self.assertIsInstance(envs, dict)
        self.assertEqual(envs[Handler.CONST_WORKER_COUNT], str(self.REPLICAS - 1))
        self.assertEqual(
            envs[Handler.CONST_JOB_ENTRYPOINT],
            PyTorchDistributedArtifact.CONST_DRIVER_SCRIPT,
        )
        self.assertEqual(envs[Handler.CONST_COMMAND], self.TORCHRUN_CMD)
        self.assertEqual(envs[cluster.OCI__RUNTIME_URI], self.TEST_REPO)
        self.assertEqual(envs[cluster.OCI__RUNTIME_GIT_COMMIT], self.TEST_COMMIT)
        self.assertEqual(envs[utils.CONST_ENV_PIP_PKG], self.PIP_PKG)
        self.assertEqual(envs[utils.CONST_ENV_PIP_REQ], self.PIP_REQ)
        self.assertEqual(
            envs[utils.CONST_ENV_INPUT_MAPPINGS],
            json.dumps({self.INPUT_SRC: self.INPUT_DST}),
        )
        self.assertNotIn(Handler.CONST_DEEPSPEED, envs)
        # Test deepspeed env var
        envs = Handler(DataScienceJob())._translate_env(
            self.init_runtime().with_command("train.py", use_deepspeed=True)
        )
        self.assertIn(Handler.CONST_DEEPSPEED, envs)

    @mock.patch("ads.jobs.builders.infrastructure.dsc_job.DSCJob.create")
    def test_extract_env(self, *args):
        """Tests extracting YAML specs from environment variables."""
        job = DataScienceJob().create(self.init_runtime())
        spec = Handler(job)._extract_envs(job.dsc_job)
        self.assertEqual(
            spec,
            {
                "conda": {"type": "service", "slug": "pytorch110_p38_gpu_v1"},
                "command": "torchrun distributed/minGPT-ddp/mingpt/main.py data_config.path=data/input.txt",
                "replicas": 2,
                "git": {
                    "url": "https://github.com/pytorch/examples.git",
                    "commit": "d91085d2181bf6342ac7dafbeee6fc0a1f64dcec",
                },
                "inputs": {"oci://bucket@namespace/path/to/input": "data/input.txt"},
                "dependencies": {
                    "pipPackages": '"package>1.0"',
                    "pipRequirements": "distributed/minGPT-ddp/requirements.txt",
                },
            },
        )

    @mock.patch("ads.jobs.builders.infrastructure.dsc_job.DSCJob.create")
    @mock.patch("ads.jobs.builders.infrastructure.dsc_job.DSCJob.run")
    def test_create_job_runs(self, patched_run, *args):
        test_ocid = "ocid-test"
        patched_run.return_value = DataScienceJobRun(id=test_ocid)
        job = DataScienceJob().create(self.init_runtime())
        runtime = self.init_runtime()
        main_run = runtime.run(job.dsc_job)
        self.assertIsInstance(main_run, DataScienceJobRun)
        self.assertEqual(main_run.id, test_ocid)
        kwarg_list = [call_args.kwargs for call_args in patched_run.call_args_list]
        self.assertEqual(
            kwarg_list,
            [
                {
                    "display_name": "None-0",
                    "environment_variables": {"RANK": "0", "WORLD_SIZE": "2"},
                },
                {
                    "display_name": "None-1",
                    "environment_variables": {
                        "RANK": "1",
                        "WORLD_SIZE": "2",
                        "MAIN_JOB_RUN_OCID": test_ocid,
                    },
                },
            ],
        )

    @mock.patch.dict(
        os.environ, {utils.CONST_ENV_INPUT_MAPPINGS: json.dumps({INPUT_SRC: INPUT_DST})}
    )
    @mock.patch("os.makedirs")
    @mock.patch("fsspec.filesystem")
    def test_copy_inputs(self, fs, makedirs):
        utils.OCIHelper.copy_inputs()
        self.assertEqual(fs.call_args.args[0], "oci")
        self.assertEqual(makedirs.call_args.args[0], "data")
