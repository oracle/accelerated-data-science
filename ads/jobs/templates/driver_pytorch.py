"""This module requires oracle-ads>=2.6.8
"""
import copy
import ipaddress
import os
import time
import subprocess
import sys

import oci
import psutil
import torch
import torch.multiprocessing as mp
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
            main_job_ocid = os.environ[CONST_ENV_MAIN_JOB_RUN_OCID]
            self.main_ip = None
        else:
            print(f"{CONST_MAIN_IP_LOG_PREFIX}{self.ip}")
            main_job_ocid = os.environ["JOB_RUN_OCID"]
            self.main_ip = self.ip
        self.main_job_run = DataScienceJobRun.from_ocid(main_job_ocid)
        self.entrypoint_env = PythonRuntimeHandler.CONST_CODE_ENTRYPOINT

    def wait_for_main_ip_address(self, timeout=15 * 60):
        second_started = time.time()
        while not self.main_ip:
            self.main_ip = self.check_ip_address()
            if self.main_ip:
                break
            if time.time() - second_started > timeout:
                raise TimeoutError(
                    f"Failed to obtain main node IP address in {timeout} seconds."
                )
            time.sleep(60)
        return self

    def check_ip_address(self):
        logs = self.main_job_run.logs()
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
                    print(f"IP Address: {ip}")
                    os.environ["GLOO_SOCKET_IFNAME"] = interface
                    os.environ["NCCL_SOCKET_IFNAME"] = interface
                    return ip
            print("IP ADDRESS NOT FOUND!!")
            return None
        else:
            import socket

            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            print(f"IP Address: {ip}")
            return ip

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

    @staticmethod
    def set_env(local_rank):
        gpu_count = torch.cuda.device_count()
        env = copy.deepcopy(os.environ)
        env["NODE_RANK"] = os.environ["RANK"]
        env["LOCAL_RANK"] = str(local_rank)
        env["RANK"] = str(int(env["NODE_RANK"]) * gpu_count + local_rank)
        return env

    @staticmethod
    def run_cmd(cmd, env=None):
        print(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}")
        if env:
            print(f"PID: {os.getpid()}, RANK: {env.get('RANK')}", flush=True)
            print(
                f"PID: {os.getpid()}, LOCAL_RANK: {env.get('LOCAL_RANK')}", flush=True
            )
        else:
            print(f"RANK: {os.environ['RANK']}", flush=True)

        training_start_time = time.time()
        ret = subprocess.run(cmd, env=env)
        if ret.returncode != 0:
            raise Exception("PyTorch distributed errored out...", ret)
        else:
            print("Training Time: ", time.time() - training_start_time, "seconds")

    @staticmethod
    def run_cmd_multi_gpu(local_rank, cmd):
        env = TorchRunner.set_env(local_rank)
        TorchRunner.run_cmd(cmd, env=env)

    def run(self):
        os.environ["MASTER_ADDR"] = self.main_ip
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29400"
        print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")

        gpu_count = torch.cuda.device_count()
        print(f"GPU COUNT: {gpu_count}")

        node_count = int(os.environ.get(OCI__WORKER_COUNT, 0)) + 1
        if gpu_count > 0:
            os.environ["WORLD_SIZE"] = str(node_count * gpu_count)
        else:
            os.environ["WORLD_SIZE"] = str(node_count)

        cmd = [sys.executable, os.environ[self.entrypoint_env]]
        cmd += sys.argv[1:]
        print("Running: ", " ".join(cmd), flush=True)
        if gpu_count > 1:
            mp.spawn(self.run_cmd_multi_gpu, args=(cmd,), nprocs=gpu_count)
        else:
            os.environ["LOCAL_RANK"] = "0"
            self.run_cmd(cmd=cmd)


def main():
    TorchRunner().fetch_code().set_working_dir().setup_python_path().wait_for_main_ip_address().run()
    driver_utils.OCIHelper.copy_outputs()


if __name__ == "__main__":
    main()
