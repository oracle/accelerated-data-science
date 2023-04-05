"""This module requires oracle-ads>=2.6.8
"""
import logging
import ipaddress
import os
import time
import socket
import sys

import oci
import psutil
import torch
from ads import set_auth
from ads.jobs import DataScienceJobRun

from ads.jobs.builders.infrastructure.dsc_job_runtime import (
    PythonRuntimeHandler,
    GitPythonRuntimeHandler,
)
from ads.opctl.distributed.common import cluster_config_helper

try:
    # This is used by ADS and testing
    from . import driver_utils
    from .driver_oci import GitSSHKey, GitManager
except ImportError:
    # This is used when the script is in a job run.
    import driver_utils
    from driver_oci import GitSSHKey, GitManager

logger = logging.getLogger(__name__)
logger = driver_utils.set_log_level(logger)


CONST_ENV_MAIN_JOB_RUN_OCID = "MAIN_JOB_RUN_OCID"
CONST_MAIN_IP_LOG_PREFIX = "Distributed Training Main IP: "
# Working count is the number of node - 1
OCI__WORKER_COUNT = "OCI__WORKER_COUNT"

set_auth("resource_principal")


class TorchRunner(driver_utils.JobRunner):
    def __init__(self, code_dir: str = driver_utils.DEFAULT_CODE_DIR) -> None:
        super().__init__(code_dir)
        self.ds_client = driver_utils.OCIHelper.init_oci_client(
            oci.data_science.DataScienceClient
        )
        self.ip = self.find_self_ip()

        if CONST_ENV_MAIN_JOB_RUN_OCID in os.environ:
            host_job_ocid = os.environ[CONST_ENV_MAIN_JOB_RUN_OCID]
            logger.debug("Host job run OCID: %s", host_job_ocid)
            self.host_ip = None
        else:
            print(f"{CONST_MAIN_IP_LOG_PREFIX}{self.ip}")
            host_job_ocid = os.environ["JOB_RUN_OCID"]
            self.host_ip = self.ip
        self.host_job_run = DataScienceJobRun.from_ocid(host_job_ocid)
        self.entrypoint_env = PythonRuntimeHandler.CONST_CODE_ENTRYPOINT
        logger.debug("Runner initialized.")

    def wait_for_main_ip_address(self, timeout=15 * 60):
        logger.info("Waiting for host's IP address...", flush=True)
        second_started = time.time()
        while not self.host_ip:
            self.host_ip = self.check_ip_address()
            if self.host_ip:
                break
            if time.time() - second_started > timeout:
                raise TimeoutError(
                    f"Failed to obtain main node IP address in {timeout} seconds."
                )
            time.sleep(60)
        logger.info("Found host IP: %s", self.host_ip)
        return self

    def check_ip_address(self):
        logger.debug("Looking for host IP...")
        logs = self.host_job_run.logs()
        for log in logs:
            if log["message"].startswith(CONST_MAIN_IP_LOG_PREFIX):
                return log["message"][len(CONST_MAIN_IP_LOG_PREFIX) :]
        return None

    def find_self_ip(self):
        """
        Identify IP address by finding which of the host IP intersects with the CIDR block of the subnet
        associated with the JOB_OCID
        """
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
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            logger.info("Node IP address: %s", ip)
            return ip

    def build_c_library(self):
        C_SOURCE_CODE = "hostname.c"
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

    def fetch_code(self):
        if GitPythonRuntimeHandler.CONST_ENTRYPOINT in os.environ:
            self.entrypoint_env = GitPythonRuntimeHandler.CONST_ENTRYPOINT
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

    def run(self):
        node_count = int(os.environ.get(OCI__WORKER_COUNT, 0)) + 1
        logger.debug("Node count: %s", node_count)

        gpu_count = torch.cuda.device_count()
        logger.debug("GPU count on this node: %s", gpu_count)

        if gpu_count > 0:
            nproc_per_node = gpu_count
        else:
            nproc_per_node = 1

        if CONST_ENV_MAIN_JOB_RUN_OCID in os.environ:
            rdzv_conf = "read_timeout=600"
            host = self.host_ip
        else:
            rdzv_conf = "is_host=1,read_timeout=600"
            host = "localhost"

        cmd = (
            f"LD_PRELOAD={self.conda_prefix}/lib/libhostname.so.1 OCI__HOSTNAME={self.ip} "
            + f"torchrun --nnode={node_count} --nproc_per_node={nproc_per_node} "
            + f"--rdzv_backend=c10d --rdzv_endpoint={host}:29400 --rdzv_conf={rdzv_conf} "
            + f"{os.environ[self.entrypoint_env]}"
        )
        args = " ".join(sys.argv[1:])
        if args:
            cmd += " ({args})"
        training_start_time = time.time()
        self.run_command(cmd, conda_prefix=self.conda_prefix, check=True)
        logger.info("Training Time: %s seconds.", time.time() - training_start_time)


def main():
    runner = (
        TorchRunner()
        .fetch_code()
        .build_c_library()
        .set_working_dir()
        .setup_python_path()
        .install_dependencies()
    )

    driver_utils.OCIHelper.copy_inputs()

    runner.wait_for_main_ip_address().run()
    driver_utils.OCIHelper.copy_outputs()


if __name__ == "__main__":
    main()
