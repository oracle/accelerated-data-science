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


CONST_ENV_MAIN_JOB_RUN_OCID = "MAIN_JOB_RUN_OCID"
CONST_ENV_LD_PRELOAD = "LD_PRELOAD"
CONST_ENV_LAUNCHER = "OCI__LAUNCHER"
CONST_MAIN_IP_LOG_PREFIX = "Distributed Training Main IP: "
# Working count is the number of node - 1
OCI__WORKER_COUNT = "OCI__WORKER_COUNT"

DEFAULT_LAUNCHER = "torchrun"

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
            time.sleep(300)
            self.host_ip = None
        else:
            print(f"{CONST_MAIN_IP_LOG_PREFIX}{self.ip}")
            host_job_ocid = os.environ["JOB_RUN_OCID"]
            self.host_ip = self.ip
        self.host_job_run = DataScienceJobRun.from_ocid(host_job_ocid)
        self.entrypoint_env = PythonRuntimeHandler.CONST_CODE_ENTRYPOINT
        self.node_count = int(os.environ.get(OCI__WORKER_COUNT, 0)) + 1
        logger.debug("Node count: %s", self.node_count)
        self.gpu_count = torch.cuda.device_count()
        logger.debug("GPU count on this node: %s", self.gpu_count)

        logger.debug("Runner initialized.")

    def wait_for_main_ip_address(self, timeout=15 * 60):
        logger.info("Waiting for host's IP address...")
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

    def _torchrun(self) -> str:
        if self.gpu_count > 0:
            nproc_per_node = self.gpu_count
        else:
            nproc_per_node = 1
        cmd = ""
        # Use LD_PRELOAD only if LD_PRELOAD is not defined by the user.
        if CONST_ENV_LD_PRELOAD not in os.environ:
            cmd = f"LD_PRELOAD={self.conda_prefix}/lib/libhostname.so.1 OCI__HOSTNAME={self.ip} "

        # The default read_timeout is 60 seconds.
        # The job run will fail if the node cannot reach the host within read_timeout.
        rdzv_timeout = os.environ.get("OCI__RDZV_TIMEOUT", "600")
        rdzv_conf = f"read_timeout={rdzv_timeout}"
        # For pytorch>=2.0, we can use f"--local_addr={self.ip} " instead of LD_PRELOAD.
        cmd += (
            f"torchrun --nnode={self.node_count} --nproc_per_node={nproc_per_node} "
            + f"--rdzv_backend=c10d --rdzv_endpoint={self.host_ip}:29400 --rdzv_conf={rdzv_conf} "
            + f"{os.environ[self.entrypoint_env]}"
        )
        return cmd

    def run(self):

        launcher_mapping = {
            "torchrun": self._torchrun
        }

        cmd = launcher_mapping[os.environ.get(CONST_ENV_LAUNCHER, DEFAULT_LAUNCHER)]()

        if sys.argv[1:]:
            cmd += " " + " ".join(shlex.quote(arg) for arg in sys.argv[1:])
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
    if METRIC_NAMESPACE:
        p = multiprocessing.Process(target=collect_metrics)
        p.daemon = True
        p.start()
    main()
