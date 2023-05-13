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
    def __init__(self, code_dir: str = driver_utils.DEFAULT_CODE_DIR) -> None:
        super().__init__(code_dir)
        self.ds_client = driver_utils.OCIHelper.init_oci_client(
            oci.data_science.DataScienceClient
        )
        self.ip = self.find_self_ip()
        self.node_ip_list = []
        self.node_runs = []

        if CONST_ENV_HOST_JOB_RUN_OCID in os.environ:
            print(f"{LOG_PREFIX_NODE_IP}{self.ip}")
            self.host_ocid = os.environ[CONST_ENV_HOST_JOB_RUN_OCID]
            logger.debug("Host job run OCID: %s", self.host_ocid)
            time.sleep(300)
            self.host_ip = None
            self.is_host = False
        else:
            print(f"{LOG_PREFIX_HOST_IP}{self.ip}")
            self.host_ocid = os.environ["JOB_RUN_OCID"]
            self.host_ip = self.ip
            self.is_host = True

        self.host_job_run = DataScienceJobRun.from_ocid(self.host_ocid)
        self.entrypoint_env = PythonRuntimeHandler.CONST_CODE_ENTRYPOINT
        self.node_count = int(os.environ.get(OCI__WORKER_COUNT, 0)) + 1
        logger.debug("Node count: %s", self.node_count)
        self.gpu_count = torch.cuda.device_count()
        logger.debug("GPU count on this node: %s", self.gpu_count)

        logger.debug("Runner initialized.")

    def wait_for_host_ip_address(self, timeout=15 * 60):
        if not self.host_ip:
            logger.info("Waiting for host's IP address...")
            self.host_ip = self.wait_for_ip_address(self.host_job_run, timeout)
        return self

    def wait_for_ip_address(self, job_run, timeout=15 * 60):
        logger.info("Waiting for IP address of job run %s", job_run.id)
        if job_run == self.host_job_run:
            log_prefix = LOG_PREFIX_HOST_IP
        else:
            log_prefix = LOG_PREFIX_NODE_IP
        ip_address = self.wait_for_log(job_run, log_prefix, timeout)
        logger.info("IP of %s: %s", job_run.id[-6:], ip_address)
        return ip_address

    def wait_for_log(self, job_run, log_prefix, timeout=15 * 60):
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

    def run(self):
        if self.gpu_count > 0:
            nproc_per_node = self.gpu_count
        else:
            nproc_per_node = 1

        cmd_prefix = self.cmd_prefix_ld_preload()
        launch_args = os.environ.get(CONST_ENV_LAUNCH_ARGS, "")
        cmd_prefix += "torchrun"
        # Add nnode, nproc_per_node and rdzv args only if they are not specified by the user.
        if "--nnode" not in launch_args:
            cmd_prefix += f" --nnode={self.node_count}"
        if "--nproc_per_node" not in launch_args:
            cmd_prefix += f" --nproc_per_node={nproc_per_node}"
        if "--rdzv_backend" not in launch_args:
            # The default read_timeout is 60 seconds.
            # The job run will fail if the node cannot reach the host within read_timeout.
            rdzv_timeout = os.environ.get("OCI__RDZV_TIMEOUT", "600")
            rdzv_conf = f"read_timeout={rdzv_timeout}"
            cmd_prefix += f" --rdzv_backend=c10d --rdzv_endpoint={self.host_ip}:29400 --rdzv_conf={rdzv_conf}"
        if launch_args:
            cmd_prefix += f" {launch_args}"
        self.run_training_script(cmd_prefix=cmd_prefix)


class DeepSpeedRunner(Runner):
    STOP_FILE = "/home/datascience/stop"
    ERROR_FILE = "/home/datascience/error"
    HOST_FILE_LOCATION = "/home/datascience/hostfile"

    def generate_key_pair(self):
        self.run_command(
            "ssh-keygen -q -t rsa -N '' <<< $'\ny'", level=logging.DEBUG, check=True
        )
        with open(os.path.join(SSH_DIR, "id_rsa.pub"), "r", encoding="utf-8") as f:
            public_key = f.read()
        print(f"{LOG_PREFIX_PUBLIC_KEY}{public_key}")
        return self

    def fetch_host_public_key(self):
        public_key = self.wait_for_log(self.host_job_run, LOG_PREFIX_PUBLIC_KEY)
        print(f"{LOG_PREFIX_PUBLIC_KEY}{public_key}")
        # logger.debug("%s", LOG_PREFIX_PUBLIC_KEY + public_key)
        auth_keys_file = os.path.join(SSH_DIR, "authorized_keys")
        os.makedirs(SSH_DIR, exist_ok=True)
        with open(auth_keys_file, "a+", encoding="utf-8") as f:
            f.write(public_key)
            f.write("\n")
        logger.debug("Host public key saved to %s", auth_keys_file)

    def generate_hostfile(self):
        runs = self.host_job_run.job.run_list()
        self.node_runs = [
            run
            for run in runs
            if run.status in ["ACCEPTED", "IN_PROGRESS"] and run.id != self.host_ocid
        ]
        self.node_ip_list = [self.wait_for_ip_address(run) for run in self.node_runs]
        logger.info("Node IPs: %s", self.node_ip_list)
        logger.debug(f"Writing hostfile to %s", self.HOST_FILE_LOCATION)
        os.makedirs(os.path.dirname(self.HOST_FILE_LOCATION), exist_ok=True)
        hostfile_content = [f"{ip} slots={self.gpu_count}" for ip in self.node_ip_list]
        with open(self.HOST_FILE_LOCATION, "w", encoding="utf-8") as f:
            f.writelines(hostfile_content)
        self.run_command(f"cat {self.HOST_FILE_LOCATION}", level=logging.DEBUG)
        ssh_config_path = os.path.join(SSH_DIR, "config")
        logger.debug("Writing SSH config to %s", ssh_config_path)
        with open(ssh_config_path, "w", encoding="utf-8") as f:
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

    def start_ssh_server(self):
        self.run_command(
            "sudo --preserve-env yum install -y openssh-server",
            level=logging.DEBUG,
            check=True,
        )
        self.run_command("sudo /usr/sbin/sshd", level=logging.DEBUG, check=True)

    def run(self):
        if self.is_host:
            self.generate_key_pair().generate_hostfile()
            # Wait for nodes to be ready
            for run in self.node_runs:
                self.wait_for_log(run, LOG_PREFIX_PUBLIC_KEY)
            for node_ip in self.node_ip_list:
                self.run_command(
                    f"ssh-keyscan -H {node_ip} >> {SSH_DIR}/known_hosts",
                    level=logging.DEBUG,
                    check=True,
                )
            # TODO: Run job here.
            # TODO: Use heartbeat to indicate job is still running.
            # Stop file will not be generated if job run is killed from the console.
            # Signal stop
            for node_ip in self.node_ip_list:
                logger.debug("Sending stop file to %s", node_ip)
                self.run_command(
                    f"ssh -v {node_ip} 'touch {self.STOP_FILE}'",
                    level=logging.DEBUG,
                    check=True,
                )
        else:
            self.run_command("sudo ssh-keygen -A", level=logging.DEBUG, check=True)
            self.start_ssh_server()
            self.fetch_host_public_key()
            while not os.path.exists(self.STOP_FILE):
                time.sleep(60)
            logger.info("Stop file found. Stopping job run...")


def main():
    launcher = os.environ.get(CONST_ENV_LAUNCHER, "torchrun").lower()
    runner_class = {"torchrun": TorchRunner, "deepspeed": DeepSpeedRunner}[launcher]
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
