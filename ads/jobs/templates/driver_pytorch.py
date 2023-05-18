#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""This module requires oracle-ads>=2.6.8
"""
import ipaddress
import logging
import multiprocessing
import os
import time
import shlex
import socket
import sys

import oci
import psutil
import torch
from ads import set_auth
from ads.jobs import DataScienceJobRun
from ads.jobs.builders.infrastructure.dsc_job_runtime import (
    PythonRuntimeHandler,
)
from ads.jobs.templates import driver_utils
from ads.opctl.distributed.common import cluster_config_helper

try:
    # This is used by ADS and testing
    from . import driver_utils
    from .driver_oci import GitSSHKey, GitManager
    from .oci_metrics import collect_metrics, METRIC_NAMESPACE
except ImportError:
    # This is used when the script is in a job run.
    import driver_utils
    from driver_oci import GitSSHKey, GitManager
    from oci_metrics import collect_metrics, METRIC_NAMESPACE

logger = logging.getLogger(__name__)
logger = driver_utils.set_log_level(logger)


CONST_ENV_HOST_JOB_RUN_OCID = "MAIN_JOB_RUN_OCID"
CONST_ENV_LD_PRELOAD = "LD_PRELOAD"
CONST_ENV_LAUNCHER = "OCI__LAUNCHER"
CONST_ENV_LAUNCH_ARGS = "OCI__LAUNCH_ARGS"
LOG_PREFIX_HOST_IP = "Distributed Training Main IP: "
LOG_PREFIX_NODE_IP = "Node IP: "
LOG_PREFIX_PUBLIC_KEY = "HOST PUBLIC KEY: "
SSH_DIR = "/home/datascience/.ssh"
# Working count is the number of node - 1
OCI__WORKER_COUNT = "OCI__WORKER_COUNT"
DEFAULT_LAUNCHER = "torchrun"

set_auth("resource_principal")


class Runner(driver_utils.JobRunner):
    """Base runner class for PyTorch training job"""

    def __init__(self, code_dir: str = driver_utils.DEFAULT_CODE_DIR) -> None:
        super().__init__(code_dir)
        self.launch_args = os.environ.get(CONST_ENV_LAUNCH_ARGS, "")
        self.ds_client = driver_utils.OCIHelper.init_oci_client(
            oci.data_science.DataScienceClient
        )
        self.ip = self.find_self_ip()
        # IP address of other nodes as a list
        self.node_ip_list = []
        # DataScienceJobRun objects of other nodes as a list
        self.node_runs = []

        if CONST_ENV_HOST_JOB_RUN_OCID in os.environ:
            # Print the node IP address to logs so that it can be obtained by the host.
            print(f"{LOG_PREFIX_NODE_IP}{self.ip}")
            self.host_ocid = os.environ[CONST_ENV_HOST_JOB_RUN_OCID]
            logger.debug("Host job run OCID: %s", self.host_ocid)
            self.host_ip = None
            self.is_host = False
        else:
            # Print the host IP address to logs so that it can be obtained by the nodes.
            print(f"{LOG_PREFIX_HOST_IP}{self.ip}")
            self.host_ocid = os.environ["JOB_RUN_OCID"]
            self.host_ip = self.ip
            self.is_host = True

        self.host_job_run = DataScienceJobRun.from_ocid(self.host_ocid)
        self.entrypoint_env = PythonRuntimeHandler.CONST_CODE_ENTRYPOINT
        # The total number of node is OCI__WORKER_COUNT + 1
        self.node_count = int(os.environ.get(OCI__WORKER_COUNT, 0)) + 1
        logger.debug("Node count: %s", self.node_count)
        self.gpu_count = torch.cuda.device_count()
        logger.debug("GPU count on this node: %s", self.gpu_count)

        logger.debug("Runner initialized.")

    def launch_args_contains(self, arg):
        return f"--{arg}" in self.launch_args

    def wait_for_host_ip_address(self, timeout=15 * 60):
        """Waits until the IP address of the host is obtained.

        Parameters
        ----------
        timeout : int, optional
            Timeout in seconds, by default 15 minutes.

        Returns
        -------
        str
            IP address
        """
        if not self.host_ip:
            logger.info("Waiting for host's IP address...")
            self.host_ip = self.wait_for_ip_address(self.host_job_run, timeout)
        return self

    def wait_for_ip_address(self, job_run, timeout=15 * 60):
        """Waits until the IP address of a particular job run is obtained.

        Parameters
        ----------
        job_run : DataScienceJobRun
            A DataScienceJobRun object
        timeout : int, optional
            Timeout in seconds, by default 15 minutes.

        Returns
        -------
        str
            IP address
        """
        logger.info("Waiting for IP address of job run %s", job_run.id)
        if job_run == self.host_job_run:
            log_prefix = LOG_PREFIX_HOST_IP
        else:
            log_prefix = LOG_PREFIX_NODE_IP
        ip_address = self.wait_for_log(job_run, log_prefix, timeout)
        logger.info("IP of %s: %s", job_run.id[-6:], ip_address)
        return ip_address

    def wait_for_log(self, job_run, log_prefix, timeout=15 * 60):
        """Waits until a log message with specific prefix is found in the logs of a job run.

        Parameters
        ----------
        job_run : DataScienceJobRun
            A DataScienceJobRun object
        log_prefix : str
            The prefix of the log message to look for.
        timeout : int, optional
            Timeout in seconds, by default 15 minutes.

        Returns
        -------
        str
            _description_

        Raises
        ------
        TimeoutError
            _description_
        """
        logger.debug(
            "Waiting for logs with prefix '%s' from %s.", log_prefix, job_run.id
        )
        second_started = time.time()
        log = None
        while not log:
            log = self.check_job_run_logs(job_run=job_run, log_prefix=log_prefix)
            if log:
                break
            if time.time() - second_started > timeout:
                raise TimeoutError(
                    f"Failed to obtain log with prefix {log_prefix} for {job_run.id} in {timeout} seconds."
                )
            time.sleep(60)
        return log

    @staticmethod
    def check_job_run_logs(job_run, log_prefix):
        logger.debug("Checking logs for job run %s", job_run.id)
        logs = job_run.logs()
        for log in logs:
            if log["message"].startswith(log_prefix):
                return log["message"][len(log_prefix) :]
        return None

    def find_self_ip(self):
        """
        Identify IP address by finding which of the host IP intersects with the CIDR block of the subnet
        associated with the JOB_OCID
        """
        hostname = socket.gethostname()
        logger.debug("Hostname: %s", hostname)
        logger.debug("Get Host by Addr: %s", socket.gethostbyaddr(socket.gethostname()))
        logger.debug("FQDN: %s", socket.getfqdn(socket.gethostname()))
        if os.environ.get("JOB_OCID"):
            subnet_id = self.ds_client.get_job(
                os.environ["JOB_OCID"]
            ).data.job_infrastructure_configuration_details.subnet_id
            core_client = driver_utils.OCIHelper.init_oci_client(
                oci.core.VirtualNetworkClient
            )
            cidr = core_client.get_subnet(subnet_id).data.cidr_block

            for interface, snics in psutil.net_if_addrs().items():
                ip = snics[0].address
                if ipaddress.ip_address(ip) in ipaddress.ip_network(cidr):
                    logger.info("Node IP address: %s", ip)
                    os.environ["GLOO_SOCKET_IFNAME"] = interface
                    os.environ["NCCL_SOCKET_IFNAME"] = interface
                    return ip
            logger.critical("Unable to determine node IP address.")
            return None
        else:
            ip = socket.gethostbyname(hostname)
            logger.info("Node IP address: %s", ip)
            return ip

    def fetch_code(self):
        if cluster_config_helper.OCI__RUNTIME_URI in os.environ:
            self._fetch_git(code_dir=self.code_dir)
        return self

    def _fetch_git(self, code_dir):
        uri = os.environ.get(cluster_config_helper.OCI__RUNTIME_URI)
        branch = os.environ.get(cluster_config_helper.OCI__RUNTIME_GIT_BRANCH)
        commit = os.environ.get(cluster_config_helper.OCI__RUNTIME_GIT_COMMIT)
        secret_ocid = os.environ.get(cluster_config_helper.OCI__RUNTIME_GIT_SECRET_ID)
        # with GitSSHKey does nothing if secret_ocid is None or empty
        with GitSSHKey(secret_ocid):
            GitManager(uri, code_dir=code_dir).fetch_repo().checkout_code(
                branch=branch, commit=commit
            )

    def run_training_script(self, cmd_prefix=""):
        cmd = os.environ[self.entrypoint_env]
        if cmd_prefix:
            cmd = cmd_prefix + " " + cmd
        if sys.argv[1:]:
            cmd += " " + " ".join(shlex.quote(arg) for arg in sys.argv[1:])
        training_start_time = time.time()
        self.run_command(cmd, conda_prefix=self.conda_prefix, check=True)
        logger.info("Training Time: %s seconds.", time.time() - training_start_time)

    def run(self):
        raise NotImplementedError()


class TorchRunner(Runner):
    RDZV_PORT = 29400

    def __init__(self, code_dir: str = driver_utils.DEFAULT_CODE_DIR) -> None:
        super().__init__(code_dir)
        self.build_c_library()

    def build_c_library(self):
        C_SOURCE_CODE = "hostname_from_env.c"
        source_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), C_SOURCE_CODE
        )
        if not os.path.exists(source_path):
            logger.error("Source code %s not found.", source_path)
            return

        self.run_command(
            "gcc -fPIC -shared -Wl,-soname,libhostname.so.1 -ldl "
            f"-o {self.conda_prefix}/lib/libhostname.so.1 {source_path}",
            conda_prefix=self.conda_prefix,
            check=True,
        )
        self.run_command(
            f"ls {self.conda_prefix}/lib/libhostname*", level=logging.DEBUG
        )

        return self

    def cmd_prefix_ld_preload(self):
        cmd_prefix = ""
        # Use LD_PRELOAD only if LD_PRELOAD is not defined by the user.
        # For pytorch>=2.0, we can use f"--local_addr={self.ip} " instead of LD_PRELOAD.
        if CONST_ENV_LD_PRELOAD not in os.environ:
            cmd_prefix = f"LD_PRELOAD={self.conda_prefix}/lib/libhostname.so.1 OCI__HOSTNAME={self.ip} "
        return cmd_prefix

    def get_rdzv_conf(self):
        # The default read_timeout is 60 seconds.
        # The job run will fail if the node cannot reach the host within read_timeout.
        rdzv_timeout = os.environ.get("OCI__RDZV_TIMEOUT", "600")
        rdzv_conf = f"read_timeout={rdzv_timeout}"
        return rdzv_conf

    def run(self):
        if self.gpu_count > 0:
            nproc_per_node = self.gpu_count
        else:
            nproc_per_node = 1

        cmd_prefix = self.cmd_prefix_ld_preload()
        cmd_prefix += "torchrun"
        # Add nnode, nproc_per_node and rdzv args only if they are not specified by the user.
        if not self.launch_args_contains("nnode"):
            cmd_prefix += f" --nnode={self.node_count}"
        if not self.launch_args_contains("nproc_per_node"):
            cmd_prefix += f" --nproc_per_node={nproc_per_node}"
        if not self.launch_args_contains("rdzv_backend"):
            cmd_prefix += f" --rdzv_backend=c10d --rdzv_endpoint={self.host_ip}:{self.RDZV_PORT} --rdzv_conf={self.get_rdzv_conf()}"
        if self.launch_args:
            cmd_prefix += f" {self.launch_args}"
        self.run_training_script(cmd_prefix=cmd_prefix)


class DeepSpeedRunner(Runner):
    STOP_FILE = "/home/datascience/stop"
    ERROR_FILE = "/home/datascience/error"
    HOST_FILE = "/home/datascience/hostfile"
    ENV_FILE = os.path.expanduser("~/.deepspeed_env")

    def generate_key_pair(self):
        self.run_command(
            "ssh-keygen -q -t rsa -N '' <<< $'\ny'", level=logging.DEBUG, check=True
        )
        with open(os.path.join(SSH_DIR, "id_rsa.pub"), "r", encoding="utf-8") as f:
            public_key = f.read()
        print(f"{LOG_PREFIX_PUBLIC_KEY}{public_key}")
        self.add_authoried_key(public_key)
        self.run_command(
            f"ssh-keyscan -H {self.host_ip} >> {SSH_DIR}/known_hosts",
            level=logging.DEBUG,
            check=True,
        )
        self.test_ssh_connection(self.host_ip)
        return self

    @staticmethod
    def add_authoried_key(public_key):
        auth_keys_file = os.path.join(SSH_DIR, "authorized_keys")
        os.makedirs(SSH_DIR, exist_ok=True)
        with open(auth_keys_file, "a+", encoding="utf-8") as f:
            f.write(public_key)
            f.write("\n")
        logger.debug("Public key saved to %s", auth_keys_file)

    def fetch_host_public_key(self):
        public_key = self.wait_for_log(self.host_job_run, LOG_PREFIX_PUBLIC_KEY)
        print(f"{LOG_PREFIX_PUBLIC_KEY}{public_key}")
        # logger.debug("%s", LOG_PREFIX_PUBLIC_KEY + public_key)
        self.add_authoried_key(public_key)

    def generate_hostfile(self):
        runs = self.host_job_run.job.run_list()
        self.node_runs = [
            run
            for run in runs
            if run.status in ["ACCEPTED", "IN_PROGRESS"] and run.id != self.host_ocid
        ]
        self.node_ip_list = [self.wait_for_ip_address(run) for run in self.node_runs]
        logger.info("Node IPs: %s", self.node_ip_list)
        # Hostfile
        logger.debug("Writing hostfile to %s", self.HOST_FILE)
        os.makedirs(os.path.dirname(self.HOST_FILE), exist_ok=True)
        host_file_content = [f"{ip} slots={self.gpu_count}" for ip in self.node_ip_list]
        with open(self.HOST_FILE, "w", encoding="utf-8") as f:
            f.write(f"{self.host_ip} slots={self.gpu_count}\n")
            f.writelines(host_file_content)
        self.run_command(f"cat {self.HOST_FILE}", level=logging.DEBUG)
        # SSH config
        ssh_config_path = os.path.join(SSH_DIR, "config")
        logger.debug("Writing SSH config to %s", ssh_config_path)
        with open(ssh_config_path, "w", encoding="utf-8") as f:
            f.writelines(
                [
                    "",
                    f"Host {self.host_ip}",
                    "IdentityFile /home/datascience/.ssh/id_rsa",
                    "User datascience",
                ]
            )
            for node_ip in self.node_ip_list:
                f.writelines(
                    [
                        "",
                        f"Host {node_ip}",
                        "IdentityFile /home/datascience/.ssh/id_rsa",
                        "User datascience",
                    ]
                )
        return self

    def test_ssh_connection(self, host):
        ret = self.run_command(
            f"ssh -v -o PasswordAuthentication=no {host} hostname -I",
            level=logging.DEBUG,
        )
        if ret == 0:
            logger.debug("SSH connection to %s - OK", host)
        else:
            logger.debug("SSH connection to %s - FAILED", host)

    def touch_file(self, filename):
        for node_ip in self.node_ip_list:
            logger.debug("Sending stop file to %s", node_ip)
            self.run_command(
                f"ssh -v {node_ip} 'touch {filename}'",
                level=logging.DEBUG,
                check=True,
            )

    def save_deepspeed_env(self):
        """Saves the environment variables for multi node training.
        DeepSpeed performs multi-node training via SSH,
        the environment variables configured by the job runs are not propagated to the SSH session.
        DeepSpeed will load the environment variables from file for the SSH sessions.
        """
        with open(self.ENV_FILE, mode="w", encoding="utf-8") as f:
            for k, v in os.environ.items():
                # As of deepspeed==0.9.2, empty value or line break will cause parsing error,
                # as the .deepspeed_env file is parsed line by line.
                if not v or "\n" in v:
                    continue
                # Quote the value if it contains space
                # Environment variable containing space may not be exported correctly when using pdsh
                # https://github.com/microsoft/DeepSpeed/blob/v0.9.2/deepspeed/launcher/multinode_runner.py#L79
                if " " in v:
                    v = shlex.quote(v)

                f.write(f"{k}={v}\n")
        logger.debug("Environment variables saved to %s", self.ENV_FILE)
        self.run_command(f"cat {self.ENV_FILE}")

    def run(self):
        # Check DeepSpeed compatibility
        self.run_command(
            "ds_report", conda_prefix=self.conda_prefix, level=logging.DEBUG
        )
        # Generate SSH host keys for SSH server
        self.run_command("sudo ssh-keygen -A", level=logging.DEBUG, check=True)
        # Install SSH server to accept SSH connections
        # DeepSpeed uses "hostname -I" to determine the IP address
        # pdsh is required for default multi node training
        # torch cpp extension uses which command to find compiler
        # DeepSpeed async_io requires libaio-devel
        self.run_command(
            "sudo --preserve-env yum install -y openssh-server hostname pdsh which libaio-devel",
            level=logging.DEBUG,
            check=True,
        )
        # Start SSH service
        self.run_command("sudo /usr/sbin/sshd", level=logging.DEBUG, check=True)
        if self.is_host:
            self.generate_key_pair().generate_hostfile()
            self.save_deepspeed_env()
            # Wait for nodes to be ready
            for run in self.node_runs:
                self.wait_for_log(run, LOG_PREFIX_PUBLIC_KEY)

            for node_ip in self.node_ip_list:
                self.run_command(
                    f"ssh-keyscan -H {node_ip} >> {SSH_DIR}/known_hosts",
                    level=logging.DEBUG,
                    check=True,
                )
            cmd_prefix = f"deepspeed --hostfile={self.HOST_FILE}"
            if self.launch_args:
                cmd_prefix += f" {self.launch_args}"
            try:
                self.run_training_script(cmd_prefix=cmd_prefix)
            except:
                # Caution: file will not be generated if job run is killed from the console.
                self.touch_file(self.ERROR_FILE)
                raise
            # Signal stop
            self.touch_file(self.STOP_FILE)
        else:
            self.fetch_host_public_key()
            # Keep the job run alive until host job run is finished.
            while not os.path.exists(self.STOP_FILE):
                time.sleep(60)
                # Stop the node if the host touched the error file.
                if os.path.exists(self.ERROR_FILE):
                    logger.error("There is an error in the host job run.")
                    sys.exit(1)
                # Stop the node if the host job run is CANCELLED or in unexpected state.
                self.host_job_run.sync()
                if self.host_job_run.status not in [
                    "ACCEPTED",
                    "IN_PROGRESS",
                    "SUCCEEDED",
                ]:
                    logger.info(
                        "Host job run status is %s. Stopping job run...",
                        self.host_job_run.status,
                    )
                    sys.exit(2)
            logger.info("Job finished successfully. Stopping job run...")


class AccelerateRunner(TorchRunner, DeepSpeedRunner):
    DEFAULT_ARGS = ["multi_gpu", "num_processes", "num_machines", "machine_rank"]
    TORCHRUN_ARGS = ["main_process_ip", "main_process_port"]

    def __init__(self, code_dir: str = driver_utils.DEFAULT_CODE_DIR) -> None:
        super().__init__(code_dir)
        self.multi_gpu = bool(self.node_count > 1 or self.gpu_count > 1)
        self.num_machines = self.node_count
        self.machine_rank = os.environ["OCI__NODE_RANK"]
        # Total number of processes across all nodes
        # Here we assume all nodes are having the same shape
        self.num_processes = (self.gpu_count if self.gpu_count else 1) * self.node_count

        self.main_process_port = self.RDZV_PORT
        self.main_process_ip = self.host_ip

    def use_deepspeed(self):
        return self.launch_args_contains("use_deepspeed")

    def accelerate_cmd(self):
        cmd = ["accelerate launch"]
        for arg in self.DEFAULT_ARGS:
            arg_val = getattr(self, arg, None)
            logger.debug("%s=%s", arg, arg_val)
            if arg_val is True:
                cmd.append(f"--{arg}")
            elif arg_val:
                cmd.extend([f"--{arg}", str(arg_val)])
        if self.launch_args:
            cmd.append(self.launch_args)
        return " ".join(cmd)

    def run_with_torchrun(self):
        cmd_prefix = self.cmd_prefix_ld_preload()
        cmd_prefix += f" {self.accelerate_cmd()}"
        for arg in self.TORCHRUN_ARGS:
            if not self.launch_args_contains(arg):
                cmd_prefix += f" --{arg} {getattr(self, arg)}"
        self.run_training_script(cmd_prefix=cmd_prefix)

    def run_with_deepspeed(self):
        raise NotImplementedError

    def run(self):
        # Check if any default argument is provided by the user
        for arg in self.DEFAULT_ARGS:
            if self.launch_args_contains(arg):
                logger.debug("%s found in launch args.", arg)
                setattr(self, arg, None)
        if self.use_deepspeed():
            self.run_with_deepspeed()
        else:
            self.run_with_torchrun()


def main():
    launcher = os.environ.get(CONST_ENV_LAUNCHER, "torchrun").lower()
    runner_class = {
        "torchrun": TorchRunner,
        "deepspeed": DeepSpeedRunner,
        "accelerate": AccelerateRunner,
    }[launcher]
    runner = runner_class()
    runner: Runner
    runner.fetch_code().set_working_dir().setup_python_path().install_dependencies()

    driver_utils.OCIHelper.copy_inputs()

    runner.wait_for_host_ip_address().run()
    driver_utils.OCIHelper.copy_outputs()


if __name__ == "__main__":
    if METRIC_NAMESPACE and torch.cuda.device_count():
        p = multiprocessing.Process(target=collect_metrics)
        p.daemon = True
        p.start()
    main()
