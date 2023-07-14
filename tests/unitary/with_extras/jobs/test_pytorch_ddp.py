#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import os
import sys
import unittest
from unittest import mock
from ads.jobs import DataScienceJobRun
from ads.jobs.builders.infrastructure.dsc_job_runtime import (
    PyTorchDistributedRuntimeHandler as Handler,
)
from ads.jobs.templates import driver_utils as utils
from ads.jobs.templates import driver_pytorch as driver


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

    @mock.patch.dict(
        os.environ, {driver.CONST_ENV_LAUNCH_CMD: "torchrun train.py --data abc"}
    )
    @mock.patch("ads.jobs.templates.driver_utils.JobRunner.run_command")
    def test_run_torchrun(self, run_command):
        runner = self.init_torch_runner()
        runner.run()
        cmd = run_command.call_args.args[0]
        self.assertTrue(cmd.startswith("LD_PRELOAD="))
        self.assertTrue(
            cmd.endswith(
                "libhostname.so.1 OCI__HOSTNAME=10.0.0.1 "
                "torchrun --nnode=1 --nproc_per_node=1 "
                "--rdzv_backend=c10d --rdzv_endpoint=10.0.0.1:29400 --rdzv_conf=read_timeout=600 "
                "train.py --data abc"
            ),
            cmd,
        )

    @mock.patch.dict(
        os.environ,
        {
            utils.CONST_ENV_PIP_PKG: "abc==1.0",
            utils.CONST_ENV_PIP_REQ: "abc/requirements.txt",
        },
    )
    @mock.patch("ads.jobs.templates.driver_utils.JobRunner.run_command")
    def test_install_deps(self, run_command):
        runner = self.init_torch_runner()
        runner.install_dependencies()
        cmd_list = [call_args.args[0] for call_args in run_command.call_args_list]
        self.assertEqual(
            cmd_list,
            [
                "pip install -r abc/requirements.txt",
                "pip install abc==1.0",
            ],
        )

    def test_run_command(self):
        runner = self.init_torch_runner()
        self.assertEqual(runner.run_command("pwd", runner.conda_prefix, check=True), 0)


class DeepSpeedRunnerTest(unittest.TestCase):
    TEST_IP = "10.0.0.1"

    def init_runner(self):
        with mock.patch("socket.gethostbyname") as GetHostIP, mock.patch(
            "ads.jobs.DataScienceJobRun.from_ocid"
        ) as GetJobRun, mock.patch(
            "ads.jobs.templates.driver_utils.JobRunner.run_command"
        ):
            GetHostIP.return_value = self.TEST_IP
            GetJobRun.return_value = DataScienceJobRun(id="ocid.abcdefghijk")
            return driver.DeepSpeedRunner()

    @mock.patch.dict(
        os.environ, {driver.CONST_ENV_LAUNCH_CMD: "deepspeed train.py --data abc"}
    )
    @mock.patch("ads.jobs.templates.driver_utils.JobRunner.run_command")
    @mock.patch("ads.jobs.templates.driver_pytorch.Runner.time_cmd")
    def test_run_single_node(self, time_cmd, *args):
        runner = self.init_runner()
        runner.run()
        self.assertEqual(time_cmd.call_args.args[0], "deepspeed train.py --data abc")

    @mock.patch("ads.jobs.templates.driver_utils.JobRunner.run_command")
    def test_touch_file(self, run_command):
        runner = self.init_runner()
        runner.node_ip_list = ["10.0.0.2", "10.0.0.3"]
        runner.touch_file("stop")
        commasnds = [call_args.args[0] for call_args in run_command.call_args_list]
        self.assertEqual(
            commasnds, ["ssh -v 10.0.0.2 'touch stop'", "ssh -v 10.0.0.3 'touch stop'"]
        )


class AccelerateRunnerTest(unittest.TestCase):
    TEST_IP = "10.0.0.1"

    def init_runner(self):
        with mock.patch(
            "ads.jobs.templates.driver_pytorch.TorchRunner.build_c_library"
        ), mock.patch("socket.gethostbyname") as GetHostIP, mock.patch(
            "ads.jobs.DataScienceJobRun.from_ocid"
        ) as GetJobRun, mock.patch(
            "ads.jobs.templates.driver_utils.JobRunner.run_command"
        ):
            GetHostIP.return_value = self.TEST_IP
            GetJobRun.return_value = DataScienceJobRun(id="ocid.abcdefghijk")
            return driver.AccelerateRunner()

    @mock.patch.dict(
        os.environ,
        {
            driver.CONST_ENV_DEEPSPEED: "1",
            driver.OCI__WORKER_COUNT: "1",
            driver.CONST_ENV_LAUNCH_CMD: "accelerate launch train.py --data abc",
            "RANK": "0",
        },
    )
    @mock.patch("ads.jobs.templates.driver_pytorch.DeepSpeedRunner.run_deepspeed_host")
    @mock.patch("ads.jobs.templates.driver_utils.JobRunner.run_command")
    @mock.patch("ads.jobs.templates.driver_pytorch.Runner.time_cmd")
    def test_run(self, time_cmd, run_command, run_deepspeed):
        run_command.return_value = 0

        runner = self.init_runner()
        runner.run_with_torchrun()
        self.assertTrue(
            time_cmd.call_args.kwargs["cmd"].endswith(
                "libhostname.so.1 OCI__HOSTNAME=10.0.0.1 "
                "accelerate launch --num_processes 2 --num_machines 2 --machine_rank 0 --main_process_port 29400 "
                "train.py --data abc"
            ),
            time_cmd.call_args.kwargs["cmd"],
        )

        runner.run()
        self.assertEqual(
            run_deepspeed.call_args.args[0],
            [
                "--num_processes",
                "2",
                "--num_machines",
                "2",
                "--machine_rank",
                "0",
                "--main_process_ip",
                "10.0.0.1",
                "--main_process_port",
                "29400",
                "--deepspeed_hostfile=/home/datascience/hostfile",
            ],
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
