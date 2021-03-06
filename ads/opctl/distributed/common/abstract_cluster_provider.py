#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import fsspec
import ads
import os
import oci
import json
from ads.jobs import Job
import ipaddress
import psutil
from time import sleep, time
import pandas as pd  # Have to find a better way for timedelta
from urllib.parse import urlparse


class ClusterProvider:
    """
    Provides contract for implementing Framework specific Cluster Life Cycle Manager.

    The main node is identified by the environment variable - `MAIN`
    The worker node is identified by the environment variable - `WORKER`

    The worker and main coordinate cluster configuration using the directory provided via `WORK_DIR`. The main node creates config.json in the `WORK_DIR`.
    The worker nodes polls the config.json and exports the configuration as environment variables
    """

    def __init__(self, mode, ephemeral=True, life_span="0h", work_dir=""):
        self.mode = mode
        self.start_time = time()
        self.ephemeral = ephemeral
        self.authinfo = self.get_oci_auth()
        self.ip = self.find_self_ip(self.authinfo)
        self.end_time = (
            pd.Timedelta(int(life_span[:-1]), life_span[-1]).total_seconds()
            + self.start_time
        )
        self.work_dir = work_dir
        self._get_my_work_dir()

        self.main_config_file = os.path.join(
            self.my_work_dir, f"MAIN_config.json"
        )  # In case of worker, we have two config file, worker generated and main config
        self.time_out = int(
            os.environ.get("OCI__TIMEOUT", "600")
        )  # Time out to wait for the config file

        self.setup_configuration()  # Write cluster configuration to `work_dir`. Eg. IP address of scheduler, etc.

        # self.tmpdir = os.environ["tmpdir"]

        self.stop_file = os.path.join(
            self.my_work_dir, "stop"
        )  # Control file to instruct cluster to stop

        scheme = urlparse(
            self.stop_file,
        ).scheme
        self.scheme = scheme

        self.execution_failure = False
        self.code_execution_complete = False

    def run_code(self):
        # Sub-class should implement this method to run code and,
        # set code_execution_complete to True after running code.
        self.code_execution_complete = True

    @property
    def stop_filesystem(self):
        authinfo = {}
        if self.scheme == "oci":
            authinfo = self.get_oci_auth()
        return fsspec.filesystem(
            self.scheme, **authinfo
        )  # FileSystem class corresponding to the URI scheme.

    def _get_my_work_dir(self):
        """
        Get the work dir subfolder for the current running job
        """
        # Use "Undefined" as job identifier when not running on OCI jobs.
        self.jobDefID = os.environ.get("JOB_OCID", "Undefined")
        # Use time as job run identifier when not running on OCI jobs
        self.jobRunID = os.environ.get("JOB_RUN_OCID", str(time()))

        self.my_work_dir = os.path.join(self.work_dir, self.jobDefID)
        self.config_file = os.path.join(
            self.my_work_dir,
            f"{self.mode}_{self.jobRunID}_config.json"
            if self.mode == "WORKER"
            else "MAIN_config.json",
        )  # Config file location

    def export_config_files(self):
        """
        By default only exports configuration generated by main. This behavior can be overridden.
        """
        return [self.main_config_file if self.mode == "WORKER" else self.config_file]

    def reached_endoflife(self):
        # TODO We can get rid of this method as JobRun takes parameter to set
        # the max run time. We dont have to handle this here.
        return (self.end_time - self.start_time) <= 0

    def get_oci_auth(self):
        profile = os.environ.get("OCI_CONFIG_PROFILE") or os.environ.get(
            "OCIFS_CONFIG_PROFILE"
        )
        ads.set_auth(
            os.environ.get("OCI_IAM_TYPE", "resource_principal"),
            profile=profile or "DEFAULT",
        )
        authinfo = ads.common.auth.default_signer()
        return authinfo

    @classmethod
    def find_self_ip(cls, authinfo):
        """
        Identify IP address by finding which of the host IP intersects with the CIDR block of the subnet associated with the JOB_OCID
        """
        if os.environ.get("JOB_OCID"):
            job_ocid = os.environ["JOB_OCID"]
            jobDef = Job.from_datascience_job(job_ocid)
            subnet_id = jobDef.infrastructure.subnet_id
            core_client = oci.core.VirtualNetworkClient(**authinfo)
            cidr = core_client.get_subnet(subnet_id).data.cidr_block
            for interface, snics in psutil.net_if_addrs().items():
                ip = snics[0].address
                if ipaddress.ip_address(ip) in ipaddress.ip_network(cidr):
                    print(f"IP address: {ip}")
                    os.environ["GLOO_SOCKET_IFNAME"] = interface
                    os.environ["NCCL_SOCKET_IFNAME"] = interface
                    return ip
            print("IP ADDRESS NOT FOUND!!")
            return None
        else:
            import socket

            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            print(f"IP address: {ip}")
            return ip

    def basic_configuration(
        self,
    ) -> dict:
        """
        Prepares basic set of configuration which is framework independent.
        This configuration is decorated later by `configuration` method implemented by framework specific implementations
        """
        config = {}
        config["OCI__MAIN_IP" if self.mode == "MAIN" else "OCI__WORKER_IP"] = self.ip
        config["tmpdir"] = self.my_work_dir
        return config

    def configuration(self, conf={}) -> dict:
        """
        Provides the configuration information of the cluster.

        conf:
            Contains generic information about the cluster, generated using `basic_configuration`. Eg. IP Address of the main process
        """
        return None

    def setup_extra_configs(self, conf: dict):
        return None

    def setup_configuration(self, config: dict = None):
        """
        Writes the configuration information into location provided by `work_dir`

        config:
            dictionary containing configuration information that needs to be shared with the workers
            if config is None, then this method calls `self.configuration` and saves the configuration
        work_dir:
            Could be any valid URI supported by fsspec

        """
        # tmpdir = self.my_work_dir
        # self.tmpdir = tmpdir
        config = config or self.configuration()
        print(f"Writing configuration: {config}", flush=True)
        with fsspec.open(self.config_file, "w", **self.authinfo) as conf:
            conf.write(json.dumps(config))

    def export_configuration(self, files):
        """
        Read the configuration in the files array  and export to environment variable
        """
        print(f"{self.mode}. Config File: {files}", flush=True)
        for file in files:
            with fsspec.open(file, **self.authinfo) as conf:
                config = json.loads(conf.read())
                print(f"Loading config: {config}", flush=True)
            for k in config:
                os.environ[k] = str(config[k])

    def expected_worker_count(
        self,
    ):
        return int(os.environ.get("OCI__WORKER_COUNT", 1))

    def fetch_all_worker_info(self):
        """
        Fetchs all the worker configs
        In some cluster the main requires information about all worker IPs apriori. This method maybe useful in such situation.
        """
        files = self.stop_filesystem.ls(
            f"{self.my_work_dir}/WORKER*config.json", refresh=True
        )
        worker_details = {}
        for file in files:
            with open(file) as wcf:
                worker_details[file] = json.loads(wcf.read())
        return worker_details

    def start_main(self):
        """
        Implement this for starting the main process. Eg. `scheduler` for Dask
        """
        pass

    def start_worker(self):
        """
        Implement this for starting the worker process. Eg. `dask-worker` for Dask
        """
        pass

    def start(self):
        """
        Starts the cluster processes
        """
        if self.mode == "MAIN":  # Check if the docker is started in scheduler mode
            print(f"Starting main process", flush=True)
            self.start_main()
        elif self.mode == "WORKER":  # Check if the docker is staretd in worker mode
            print(f"Starting worker process", flush=True)
            self.when_ready(self.start_worker)
        else:
            print(f"Not a valid mode: {self.mode}", flush=True)
            raise Exception("Not a valid mode")

    def check_cluster_status(self):
        pass

    def execution_failed(self):
        """
        Invoked when code submitted to epheramal cluster fails. Calling this method sets the cluster tearable state
        """
        self.execution_failure = True

    def tearable(self):
        """
        Checks if cluster is ready for tear down.
        If there is a `stop` file in the tmp directory then stop.
        If cluster is ephemeral type, check if code execution is complete
        If TTL is reached then stop
        """

        if self.stop_filesystem.exists(self.stop_file) or self.execution_failure:
            print(
                f"Stopping the process. Reason: {'Code Execution Failure' if self.execution_failure else 'Stop file found'}",
                flush=True,
            )
            return True

        if self.ephemeral:
            return self.code_execution_complete
        else:
            return self.reached_endoflife()

    def stop(self):
        """
        Writes stop file and exits.
        """
        if not self.stop_filesystem.exists(self.stop_file):
            print("Stop file not found. Writing stop file....", flush=True)
            authinfo = {}
            if self.scheme == "oci":
                authinfo = self.get_oci_auth()
            with fsspec.open(self.stop_file, "w", **authinfo) as sf:
                sf.write("stop")

    def ready(self):
        if self.stop_filesystem.exists(self.main_config_file):
            return True
        else:
            return False

    def when_ready(self, func, *args, **kwargs):
        start_time = time()
        while not self.ready():
            sleep(10)
            if time() - start_time > self.time_out:
                raise Exception(
                    "Timed out waiting to be in ready status. Likely cause configuration is missing in the `WORK_DIR`"
                )

        func(*args, **kwargs)
