import json
import unittest
import zipfile
from ads.jobs import PyTorchDistributedRuntime, DataScienceJob
from ads.jobs.builders.infrastructure.dsc_job_runtime import (
    PyTorchDistributedRuntimeHandler as Handler,
)
from ads.jobs.builders.runtimes.pytorch_runtime import (
    PyTorchDistributedArtifact,
    GitPythonArtifact,
)
from ads.opctl.distributed.common import cluster_config_helper as Cluster
from ads.jobs.templates import driver_utils as Driver


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
        envs = Handler(DataScienceJob())._translate_env(self.init_runtime())
        self.assertIsInstance(envs, dict)
        self.assertEqual(envs[Handler.CONST_WORKER_COUNT], str(self.REPLICAS - 1))
        self.assertEqual(
            envs[Handler.CONST_JOB_ENTRYPOINT],
            PyTorchDistributedArtifact.CONST_DRIVER_SCRIPT,
        )
        self.assertEqual(envs[Handler.CONST_COMMAND], self.TORCHRUN_CMD)
        self.assertEqual(envs[Cluster.OCI__RUNTIME_URI], self.TEST_REPO)
        self.assertEqual(envs[Cluster.OCI__RUNTIME_GIT_COMMIT], self.TEST_COMMIT)
        self.assertEqual(envs[Driver.CONST_ENV_PIP_PKG], self.PIP_PKG)
        self.assertEqual(envs[Driver.CONST_ENV_PIP_REQ], self.PIP_REQ)
        self.assertEqual(
            envs[Driver.CONST_ENV_INPUT_MAPPINGS],
            json.dumps({self.INPUT_SRC: self.INPUT_DST}),
        )
        self.assertNotIn(Handler.CONST_DEEPSPEED, envs)
