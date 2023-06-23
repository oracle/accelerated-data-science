import json
import os
import sys
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
from ads.jobs.templates import driver_pytorch as driver


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


class PyTorchRunnerTest(unittest.TestCase):
    TEST_IP = "10.0.0.1"
    TEST_HOST_IP = "10.0.0.100"
    TEST_HOST_OCID = "ocid_host"
    TEST_NODE_OCID = "ocid_node"

    def init_torch_runner(self):
        with mock.patch(
            "ads.jobs.templates.driver_pytorch.TorchRunner.build_c_library"
        ), mock.patch("socket.gethostbyname") as GetHostIP, mock.patch(
            "ads.jobs.DataScienceJobRun.from_ocid"
        ) as GetJobRun:
            GetHostIP.return_value = self.TEST_IP
            GetJobRun.return_value = DataScienceJobRun(id="ocid.abcdefghijk")
            return driver.TorchRunner()

    @mock.patch.dict(os.environ, {driver.CONST_ENV_HOST_JOB_RUN_OCID: TEST_HOST_OCID})
    def test_init_torch_runner_at_node(self):
        runner = self.init_torch_runner()
        self.assertEqual(runner.host_ocid, self.TEST_HOST_OCID)
        self.assertEqual(runner.host_ip, None)

    @mock.patch.dict(os.environ, {driver.CONST_ENV_JOB_RUN_OCID: TEST_NODE_OCID})
    def test_init_torch_runner_at_host(self):
        runner = self.init_torch_runner()
        self.assertEqual(runner.host_ocid, self.TEST_NODE_OCID)
        self.assertEqual(runner.host_ip, self.TEST_IP)

    @mock.patch.dict(os.environ, {driver.CONST_ENV_HOST_JOB_RUN_OCID: TEST_HOST_OCID})
    def test_wait_for_host_ip(self):
        with mock.patch("ads.jobs.DataScienceJobRun.logs") as get_logs:
            get_logs.return_value = [
                {"message": f"{driver.LOG_PREFIX_HOST_IP} {self.TEST_HOST_IP}"}
            ]
            runner = self.init_torch_runner()
            self.assertEqual(runner.host_ip, None)
            runner.wait_for_host_ip_address()
            self.assertEqual(runner.host_ip, self.TEST_HOST_IP)

    @mock.patch.dict(
        os.environ, {driver.CONST_ENV_LAUNCH_CMD: "torchrun train.py --data abc"}
    )
    def test_launch_cmd(self):
        runner = self.init_torch_runner()
        self.assertTrue(runner.launch_cmd_contains("data"))
        self.assertFalse(runner.launch_cmd_contains("data1"))
        self.assertEqual(
            runner.prepare_cmd(prefix="A=1"), "A=1 torchrun train.py --data abc"
        )

    @mock.patch.dict(os.environ, {Handler.CONST_CODE_ENTRYPOINT: "train.py"})
    @mock.patch.object(sys, "argv", ["python", "hello", "--data", "abc"])
    def test_prepare_cmd_with_entrypoint_args(self):
        runner = self.init_torch_runner()
        self.assertEqual(
            runner.prepare_cmd(launch_args=["--key", "val"], prefix="A=1"),
            "A=1 torchrun --key val train.py hello --data abc",
        )


class LazyEvaluateTest(unittest.TestCase):
    def test_lazy_evaluation(self):
        def func(a, b):
            return a + b

        def func_with_error():
            raise ValueError()

        lazy_val = driver.LazyEvaluate(func, 1, 1)
        self.assertEqual(str(lazy_val), "2")

        lazy_val = driver.LazyEvaluate(func_with_error)
        self.assertEqual(str(lazy_val), "ERROR: ")
