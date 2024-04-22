#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""This module requires oracle-ads>=2.6.8
"""
import getpass
import ipaddress
import logging
import multiprocessing
import os
import time
import shlex
import socket
import sys
import traceback

import oci
import psutil
import torch
from ads import set_auth
from ads.jobs import DataScienceJobRun
from ads.jobs.builders.infrastructure.dsc_job_runtime import (
    PythonRuntimeHandler,
)
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


# Envs provisioned by the service
CONST_ENV_HOST_JOB_RUN_OCID = "MAIN_JOB_RUN_OCID"
CONST_ENV_JOB_RUN_OCID = "JOB_RUN_OCID"
# Envs set by the ADS API
OCI__WORKER_COUNT = "OCI__WORKER_COUNT"
CONST_ENV_NODE_RANK = "NODE_RANK"
CONST_ENV_NODE_COUNT = "NODE_COUNT"
CONST_ENV_LAUNCH_CMD = "OCI__LAUNCH_CMD"
CONST_ENV_DEEPSPEED = "OCI__DEEPSPEED"
# Envs set by this module
CONST_ENV_WORLD_SIZE = "WORLD_SIZE"
CONST_ENV_LD_PRELOAD = "LD_PRELOAD"
# Envs for debugging only
# OCI_ODSC_SERVICE_ENDPOINT is used for all processes in the job run
CONST_ENV_ODSC_SERVICE_ENDPOINT = "OCI_ODSC_SERVICE_ENDPOINT"
# OCI_DS_SERVICE_ENDPOINT is used only by the training process
CONST_ENV_DS_SERVICE_ENDPOINT = "OCI_DS_SERVICE_ENDPOINT"

# Constants used in logs
LOG_PREFIX_HOST_IP = "Distributed Training HOST IP: "
LOG_PREFIX_NODE_IP = "Node IP: "
LOG_PREFIX_PUBLIC_KEY = "HOST PUBLIC KEY: "
# Other constants used within this script
# Other constants used within this script
USER_HOME = os.environ.get("HOME", f"/home/{getpass.getuser()}")
SSH_DIR = os.environ.get("OCI__SSH_DIR", os.path.join(USER_HOME, ".ssh"))
DEFAULT_LAUNCHER = "torchrun"

# Set authentication method to resource principal
# This script is expected to be running inside the job run
if "OCI_RESOURCE_PRINCIPAL_VERSION" in os.environ:
    set_auth("resource_principal")


class LazyEvaluate:
    """This is a class to delay the function call until
    its return value is needed for logging purpose.

    Example::
        logger.debug("The value is %s", LazyEvaluate(the_function, *args, **kwargs))

    Python logging will only call the __str__() method when the value is needed.

    In the above example, if the log level is INFO or above,
    the_function() will not be called/evaluated.
    If the log level is DEBUG, the_function will be called,
    and if there is an error, the error will be logged.
    The program will continue to run even if the error happens during logging.

    """

    def __init__(self, func, *args, **kwargs) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def eval(self):
        """Evaluates the function call."""
        return self.func(*self.args, **self.kwargs)

    def __str__(self) -> str:
        """Evaluate the function call and convert the return value as a string."""
        try:
            val = str(self.eval())
        except Exception as ex:
            logger.debug(traceback.format_exc())
            val = f"ERROR: {str(ex)}"
        return val


class Runner(driver_utils.JobRunner):
    """Base runner class for PyTorch training job"""

    # LAUNCHER stores the main command for launching the training job.
    # e.g. torchrun, deepspeed, accelerate, etc.
    LAUNCHER = ""

    def __init__(self, code_dir: str = driver_utils.DEFAULT_CODE_DIR) -> None:
        super().__init__(code_dir)
        self.launch_cmd = os.environ.get(CONST_ENV_LAUNCH_CMD, "")

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
            self.host_ocid = os.environ.get(CONST_ENV_JOB_RUN_OCID)
            self.host_ip = self.ip
            self.is_host = True

        self.host_job_run = DataScienceJobRun.from_ocid(self.host_ocid)
        self.entrypoint_env = PythonRuntimeHandler.CONST_CODE_ENTRYPOINT
        # The total number of nodes is OCI__WORKER_COUNT + 1
        if CONST_ENV_NODE_COUNT in os.environ:
            self.node_count = int(os.environ[CONST_ENV_NODE_COUNT])
        else:
            self.node_count = int(os.environ.get(OCI__WORKER_COUNT, 0)) + 1
        logger.debug("Node count: %s", self.node_count)
        self.gpu_count = torch.cuda.device_count()
        logger.debug("GPU count on this node: %s", self.gpu_count)

        logger.debug("Runner initialized.")

    def launch_cmd_contains(self, arg) -> bool:
        """Checks if the cmd for launching the training contains specific keyword argument."""
        return f"--{arg}" in self.launch_cmd

    def wait_for_host_ip_address(self, timeout=15 * 60) -> str:
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

    def wait_for_ip_address(self, job_run, timeout=15 * 60) -> str:
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
        ip_address = self.wait_for_log(job_run, log_prefix, timeout).strip()
        logger.info("IP of %s: %s", job_run.id[-6:], ip_address)
        return ip_address

    def wait_for_log(self, job_run, log_prefix, timeout=15 * 60) -> str:
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
            The log message with out the prefix.

        Raises
        ------
        TimeoutError
            Failed to obtain the log message within the specific timeout.
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
    def check_job_run_logs(job_run, log_prefix: str) -> str:
        """Checks the logs of a specific job run and find the log message with specific prefix.

        Parameters
        ----------
        job_run : DataScienceJobRun
            The Job run object from which the logs will be obtained.
        log_prefix : str
            The prefix to look for.

        Returns
        -------
        str
            The log message without the prefix.
        """
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
        logger.debug(
            "Get Host by Addr: %s", LazyEvaluate(socket.gethostbyaddr, hostname)
        )
        logger.debug("FQDN: %s", LazyEvaluate(socket.getfqdn, hostname))
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
                    # Specify the network interface for NCCL/GLOO
                    os.environ["GLOO_SOCKET_IFNAME"] = interface
                    os.environ["NCCL_SOCKET_IFNAME"] = interface
                    return ip
            raise EnvironmentError("Unable to determine node IP address.")
        else:
            ip = socket.gethostbyname(hostname)
            logger.info("Node IP address: %s", ip)
            return ip

    def fetch_code(self):
        """Fetches source code from Git if repo uri is specified."""
        if cluster_config_helper.OCI__RUNTIME_URI in os.environ:
            self._fetch_git(code_dir=self.code_dir)
        return self

    def _fetch_git(self, code_dir):
        """Fetches source code from Git repository."""
        uri = os.environ.get(cluster_config_helper.OCI__RUNTIME_URI)
        branch = os.environ.get(cluster_config_helper.OCI__RUNTIME_GIT_BRANCH)
        commit = os.environ.get(cluster_config_helper.OCI__RUNTIME_GIT_COMMIT)
        secret_ocid = os.environ.get(cluster_config_helper.OCI__RUNTIME_GIT_SECRET_ID)
        # with GitSSHKey does nothing if secret_ocid is None or empty
        with GitSSHKey(secret_ocid):
            GitManager(uri, code_dir=code_dir).fetch_repo().checkout_code(
                branch=branch, commit=commit
            )

    def get_cmd_with_entrypoint_and_args(self, prefix: str = "") -> str:
        """Gets the command based on entrypoint and arguments.

        Parameters
        ----------
        prefix : str, optional
            Command prefix, by default ""
            This can be used to set environment variables for the command.
            e.g. ENV=1 command

        Returns
        -------
        str
            The command including the prefix, entrypoint and arguments.
        """
        cmd = os.environ[self.entrypoint_env]
        if prefix:
            cmd = prefix + " " + cmd
        if sys.argv[1:]:
            cmd += " " + " ".join(sys.argv[1:])
        return cmd

    def prepare_cmd(self, launch_args: list = None, prefix=""):
        """Prepares the command for starting the training.

        Parameters
        ----------
        launch_args : list
            The command and arguments for starting the training as a list.
        prefix : str, optional
            The prefix to be added to the launch_args in the command, by default ""
            This can be used to set environment variables for the command.
            e.g. ENV=1 command

        Returns
        -------
        str
            The command for starting the training.
        """
        if not launch_args:
            launch_args = []
        # Append launch cmd args specified by the user.
        if self.launch_cmd:
            if self.LAUNCHER:
                if not self.launch_cmd.startswith(self.LAUNCHER):
                    raise ValueError(f"Command not supported: '{self.launch_cmd}'. ")

                launch_args.append(self.launch_cmd[len(self.LAUNCHER) + 1 :])
            else:
                launch_args.append(self.launch_cmd)
        else:
            launch_args.append(self.get_cmd_with_entrypoint_and_args())

        if prefix:
            launcher = f"{prefix} {self.LAUNCHER}"
        else:
            launcher = self.LAUNCHER

        return f"{launcher} {' '.join(launch_args)}"

    def time_cmd(self, cmd):
        """Run the command and log the time used."""
        # Show current working directory for debugging purpose
        self.run_command("pwd", level=logging.DEBUG)
        # Show all environment variables
        self.run_command("printenv", level=logging.DEBUG)
        training_start_time = time.time()
        self.run_command(cmd, conda_prefix=self.conda_prefix, check=True)
        logger.info("Time: %s seconds.", time.time() - training_start_time)

    def run(self):
        raise NotImplementedError()


class TorchRunner(Runner):
    RDZV_PORT = 29400
    LAUNCHER = "torchrun"

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
            level=logging.DEBUG,
        )
        self.run_command(
            f"ls {self.conda_prefix}/lib/libhostname*", level=logging.DEBUG
        )

        return self

    def env_ld_preload(self) -> str:
        """Generate environment variable config for LD_PRELOAD and OCI__HOSTNAME.
        The return value can be used as the prefix of a bash command.
        """
        cmd_prefix = ""
        # Use LD_PRELOAD only if LD_PRELOAD is not defined by the user.
        # For pytorch>=2.0, we can use f"--local_addr={self.ip} " instead of LD_PRELOAD.
        if CONST_ENV_LD_PRELOAD not in os.environ:
            cmd_prefix = f"LD_PRELOAD={self.conda_prefix}/lib/libhostname.so.1 OCI__HOSTNAME={self.ip}"
        return cmd_prefix

    def get_rdzv_conf(self) -> str:
        """Prepare additional rendezvous config for torch run.

        The default read_timeout is 60 seconds.
        The job run will fail if the node cannot reach the host within read_timeout.
        """
        rdzv_timeout = os.environ.get("OCI__RDZV_TIMEOUT", "600")
        rdzv_conf = f"read_timeout={rdzv_timeout}"
        return rdzv_conf

    def run(self):
        if self.gpu_count > 0:
            nproc_per_node = self.gpu_count
        else:
            nproc_per_node = 1

        launch_args = []
        # Add nnode, nproc_per_node and rdzv args only if they are not specified by the user.
        if not self.launch_cmd_contains("nnode"):
            launch_args.append(f"--nnode={self.node_count}")
        if not self.launch_cmd_contains("nproc_per_node"):
            launch_args.append(f"--nproc_per_node={nproc_per_node}")
        if not self.launch_cmd_contains("rdzv_backend"):
            launch_args.extend(
                [
                    "--rdzv_backend=c10d",
                    f"--rdzv_endpoint={self.host_ip}:{self.RDZV_PORT}",
                    f"--rdzv_conf={self.get_rdzv_conf()}",
                ]
            )

        self.time_cmd(cmd=self.prepare_cmd(launch_args, prefix=self.env_ld_preload()))


class DeepSpeedRunner(Runner):
    STOP_FILE = "/home/datascience/stop"
    ERROR_FILE = "/home/datascience/error"
    HOST_FILE = "/home/datascience/hostfile"
    ENV_FILE = os.path.expanduser("~/.deepspeed_env")
    LAUNCHER = "deepspeed"

    def __init__(self, code_dir: str = driver_utils.DEFAULT_CODE_DIR) -> None:
        super().__init__(code_dir)
        self.update_os()

    def update_os(self):
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
        # Check DeepSpeed compatibility
        self.run_command(
            "ds_report", conda_prefix=self.conda_prefix, level=logging.DEBUG
        )
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
        """Creates an empty file with specific name on all the worker nodes."""
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
                # Ignore variables that are node specific
                # The network interface name for each job run is a unique string, e.g. ens300f0v1604
                if k in ["NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "JOB_RUN_OCID"]:
                    continue
                # Quote the value if it contains space
                # Environment variable containing space may not be exported correctly when using pdsh
                # https://github.com/microsoft/DeepSpeed/blob/v0.9.2/deepspeed/launcher/multinode_runner.py#L79
                if " " in v:
                    v = shlex.quote(v)

                f.write(f"{k}={v}\n")
            # The following are required for specifying the network interface to be used by NCCL/GLOO
            # The value should be the prefix of the expected network interface name
            # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname
            f.write("NCCL_SOCKET_IFNAME=ens\n")
            f.write("GLOO_SOCKET_IFNAME=ens\n")
        logger.debug("Environment variables saved to %s", self.ENV_FILE)
        self.run_command(f"cat {self.ENV_FILE}")

    def run_deepspeed_host(self, launch_args=None):
        """Prepares the host and launch the deepspeed training.

        Parameters
        ----------
        launch_args : str, optional
            Additional command line arguments, by default None.
            The deepspeed host file should be specified in the launch args.
            For "deepspeed": --hostfile
            For "accelerate launch": --deepspeed_hostfile
        """
        if self.node_count > 1:
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

        cmd = self.prepare_cmd(launch_args)
        # For DeepSpeed, we only need to run the cmd on the host
        try:
            self.time_cmd(cmd)
        except:
            # Caution: file will not be generated if job run is killed from the console.
            self.touch_file(self.ERROR_FILE)
            raise
        # Signal stop
        self.touch_file(self.STOP_FILE)

    def run_deepspeed_worker(self):
        self.fetch_host_public_key()
        # Keep the job run alive until host job run is finished.
        while not os.path.exists(self.STOP_FILE):
            time.sleep(60)
            # Stop the node if the host touched the error file.
            if os.path.exists(self.ERROR_FILE):
                logger.error("There is an error in the host job run.")
                sys.exit(1)
            # Stop the node if the host job run is CANCELLED or in unexpected state.
            try:
                self.host_job_run.sync()
            except oci.exceptions.TransientServiceError:
                # Ignore the transient error and try again next time.
                continue
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

    def run(self):
        if self.is_host:
            if self.node_count > 1:
                launch_args = [f"--hostfile={self.HOST_FILE}"]
            else:
                launch_args = []
            self.run_deepspeed_host(launch_args)
        else:
            self.run_deepspeed_worker()


class GenericRunner(TorchRunner, DeepSpeedRunner):
    """Runner for running command other than ``torchrun``, ``deepspeed`` or ``accelerate``."""

    LAUNCHER = ""

    def use_deepspeed(self) -> bool:
        """Indicate if DeepSpeed is used."""
        if os.environ.get(CONST_ENV_DEEPSPEED):
            return True
        return False

    def set_env_var(self):
        """Set default environment variables."""
        defaults = {
            "WORLD_SIZE": self.node_count * self.gpu_count,
            "MASTER_ADDR": self.host_ip,
            "MASTER_PORT": self.RDZV_PORT,
        }
        for k, v in defaults.items():
            if k not in os.environ:
                os.environ[k] = str(v)

    def run(self):
        """Runs the user's command.
        Note that for TorchRunner or DeepSpeedRunner,
        we automatically add arguments for some settings,
        like the number of nodes and the host node address.

        This generic runner does not modify the command specified by the user.
        User needs to make sure the command can work on all nodes.
        User may use the environment variables in the command.
        """
        self.set_env_var()
        if self.use_deepspeed():
            if self.is_host:
                self.run_deepspeed_host()
            else:
                self.run_deepspeed_worker()
        else:
            self.time_cmd(cmd=self.prepare_cmd(prefix=self.env_ld_preload()))


class AccelerateRunner(TorchRunner, DeepSpeedRunner):
    """Runner for HuggingFace Accelerate."""

    # accelerate launch will add main_process_port for deepspeed cmd even if it is not needed.
    # https://github.com/huggingface/accelerate/blob/70920895e80f78d96d8f91e0beeb3ebdb8e5e5d6/src/accelerate/utils/launch.py#L233
    DEFAULT_ARGS = [
        "num_processes",
        "num_machines",
        "machine_rank",
        "main_process_ip",
        "main_process_port",
    ]
    TORCHRUN_ARGS = []
    LAUNCHER = "accelerate launch"

    def __init__(self, code_dir: str = driver_utils.DEFAULT_CODE_DIR) -> None:
        super().__init__(code_dir)
        # For "accelerate launch", only one of the following options can be used at one time
        # `--cpu`, `--multi_gpu`, `--tpu`, `--use_deepspeed`, `--use_fsdp`.
        # When a config file is not provided,
        # --multi_gpu will be set automatically if there is more than 1 GPU
        # self.multi_gpu = bool(self.node_count > 1 or self.gpu_count > 1)
        self.num_machines = self.node_count
        self.machine_rank = os.environ["NODE_RANK"]
        # Total number of processes across all nodes
        # Here we assume all nodes are having the same shape
        self.num_processes = (self.gpu_count if self.gpu_count else 1) * self.node_count

        self.main_process_port = self.RDZV_PORT
        # Host IP is not ready at initialization
        self.main_process_ip = None

    def use_deepspeed(self):
        """Indicate if DeepSpeed is used."""
        # Accelerate support using DeepSpeed by adding the "--use_deepspeed" argument.
        if os.environ.get(CONST_ENV_DEEPSPEED) or self.launch_cmd_contains(
            "use_deepspeed"
        ):
            return True
        return False

    def accelerate_args(self):
        """Gets the default arguments for the accelerate command.
        The value of the default arguments are assigned in ``__init__()``.
        """
        args = []
        for arg in self.DEFAULT_ARGS:
            arg_val = getattr(self, arg, None)
            logger.debug("%s=%s", arg, arg_val)
            if arg_val is True:
                args.append(f"--{arg}")
            elif arg_val:
                args.extend([f"--{arg}", str(arg_val)])
        return args

    def run_with_torchrun(self):
        """Runs the job with torchrun."""
        launch_args = self.accelerate_args()
        for arg in self.TORCHRUN_ARGS:
            if not self.launch_cmd_contains(arg):
                launch_args.extend([f"--{arg}", f"{getattr(self, arg)}"])
        cmd = self.prepare_cmd(launch_args, prefix=self.env_ld_preload())
        self.time_cmd(cmd=cmd)

    def run_with_deepspeed(self):
        """Runs the job with DeepSpeed."""
        if self.is_host:
            launch_args = self.accelerate_args()
            if self.num_machines > 1:
                launch_args.append(f"--deepspeed_hostfile={self.HOST_FILE}")
            self.run_deepspeed_host(launch_args)
        else:
            self.run_deepspeed_worker()

    def run(self):
        self.main_process_ip = self.host_ip
        # Check if any default argument is provided by the user
        for arg in self.DEFAULT_ARGS:
            if self.launch_cmd_contains(arg):
                logger.debug("%s found in command.", arg)
                setattr(self, arg, None)
        if self.use_deepspeed():
            self.run_with_deepspeed()
        else:
            self.run_with_torchrun()


def main():
    launch_cmd = os.environ.get(CONST_ENV_LAUNCH_CMD)
    if not launch_cmd or launch_cmd.startswith("torchrun "):
        # Use torchrun as default if launch cmd is not provided
        runner_class = TorchRunner
    elif launch_cmd.startswith("deepspeed "):
        runner_class = DeepSpeedRunner
    elif launch_cmd.startswith("accelerate "):
        runner_class = AccelerateRunner
    else:
        runner_class = GenericRunner

    runner = runner_class()
    runner: Runner
    runner.fetch_code().set_working_dir().setup_python_path().install_dependencies()

    driver_utils.OCIHelper.copy_inputs()

    runner.wait_for_host_ip_address().run()
    driver_utils.OCIHelper.copy_outputs()


if __name__ == "__main__":
    # Collect GPU metrics only if GPU is available and user defined METRIC_NAMESPACE
    if METRIC_NAMESPACE and torch.cuda.device_count():
        p = multiprocessing.Process(target=collect_metrics)
        p.daemon = True
        p.start()
    main()
