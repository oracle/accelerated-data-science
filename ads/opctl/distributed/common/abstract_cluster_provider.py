#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import ipaddress
import json
import os
import sys
from time import sleep, time, time_ns
from urllib.parse import urlparse
import subprocess


import ads
import fsspec
import oci
import pandas as pd  # Have to find a better way for timedelta
import psutil
from ads.jobs import Job
from ads.opctl.distributed.common import cluster_config_helper
from ads.jobs.templates.driver_oci import GitSSHKey, GitManager
from ads.jobs.templates.driver_utils import JobRunner
from ads.jobs.builders.runtimes.artifact import Artifact


class ClusterProvider:
    """
    Provides contract for implementing Framework specific Cluster Life Cycle Manager.

    The main node is identified by the environment variable - `MAIN`
    The worker node is identified by the environment variable - `WORKER`

    The worker and main coordinate cluster configuration using the directory provided via `WORK_DIR`. The main node creates config.json in the `WORK_DIR`.
    The worker nodes polls the config.json and exports the configuration as environment variables
    """

    SYNC_SCRIPT_PATH = "/etc/datascience/sync.sh"

    DEFAULT_CODE_DIR = "/code"

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

        self.fetch_code()

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
        self.sync()

    def sync(self, loop=True):
        sync_artifacts = os.environ.get("SYNC_ARTIFACTS", 0)
        print(f"sync_artifacts - {sync_artifacts}")
        if sync_artifacts == "1":
            bkt_name, prefix = self.get_sync_loc()
            if bkt_name is None:
                print(
                    "WARNING: Sync 'WORKSPACE', 'WORKSPACE_PREFIX' or 'work_dir' not configured. Skipping Sync"
                )
                return
            sync_script_fn = self.SYNC_SCRIPT_PATH
            if not os.path.exists(sync_script_fn):
                sync_script = self.get_sync_script()
                self.create_sync_script(sync_script_fn, sync_script)
            subprocess.Popen(
                [sync_script_fn, bkt_name, prefix, "-l"]
            ) if loop else subprocess.Popen([sync_script_fn, bkt_name, prefix])
        if not loop:
            sleep_duration = int(os.environ.get("POST_PROCESSING_WAIT", 60))
            print(f"post processing wait..{sleep_duration} seconds")
            sleep(sleep_duration)
            print("..")

    def get_sync_loc(self):
        bckt_name = os.environ.get("WORKSPACE")
        pfx = os.environ.get("WORKSPACE_PREFIX")
        if bckt_name is None:
            scheme = urlparse(self.work_dir)
            if scheme.scheme == "oci":
                bckt_name = scheme.netloc.split("@")[0]
                pfx = scheme.path
                pfx = pfx.strip("//")
        return bckt_name, pfx

    def profile_cmd(self):
        profile = os.environ.get("PROFILE", "0")
        cmd = []
        if profile == "1":
            print("Profiler ON")
            cmd = os.environ.get("PROFILE_CMD").split(" ")
        return cmd

    @staticmethod
    def create_sync_script(sync_script_fn, sync_script):
        sync_script_fn_obj = open(sync_script_fn, "w")
        sync_script_fn_obj.write(sync_script)
        sync_script_fn_obj.close()
        os.chmod(sync_script_fn, 755)

    def get_sync_script(self):
        return sync_script_str

    def fetch_code(self):
        runtime_type = os.environ.get(cluster_config_helper.OCI__RUNTIME_TYPE)
        if not runtime_type:
            return
        delegates = {"git": self._fetch_git, "remote": self._fetch_remote}
        if runtime_type not in delegates:
            raise ValueError(f"Runtime type {runtime_type} is not supported.")
        if cluster_config_helper.OCI__CODE_DIR not in os.environ:
            os.environ[cluster_config_helper.OCI__CODE_DIR] = self.DEFAULT_CODE_DIR
        delegates[runtime_type](
            code_dir=os.environ.get(cluster_config_helper.OCI__CODE_DIR)
        )

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
        JobRunner(code_dir=code_dir).setup_python_path(
            python_path=os.environ.get(cluster_config_helper.OCI__RUNTIME_PYTHON_PATH),
        )

    def _fetch_remote(self, code_dir):
        uri = os.environ.get(cluster_config_helper.OCI__RUNTIME_URI)
        Artifact.copy_from_uri(uri, code_dir)

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
        self.jobRunID = os.environ.get("JOB_RUN_OCID", str(time_ns()))

        self.my_work_dir = os.path.join(self.work_dir, self.jobDefID)
        self.config_file = os.path.join(
            self.my_work_dir,
            "MAIN_config.json"
            if self.mode == "MAIN"
            else f"{self.mode}_{self.jobRunID}_config.json",
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
        print(
            f"Writing configuration: {config} to file: {self.config_file}", flush=True
        )
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

    def start_ps(self):
        """
        Implement this for starting the ps process. Eg. `tf-parameter-server` for tensorflow
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
        elif self.mode == "PS":  # Check if the docker is staretd in worker mode
            print(f"Starting ps process", flush=True)
            self.when_ready(self.start_ps)
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


sync_script_str = """#!/bin/bash

loop=false

if [ "$3" == "-l" ]; then
  loop=true;
fi
echo "loop: $loop"
sleep_duration=${SYNC_SLEEP:-60}
if [ "$OCI_IAM_TYPE" = 'api_key' ]; then
  auth_method=api_key
else
  auth_method=resource_principal
fi
echo "auth method: $auth_method"
echo "OCI__SYNC_DIR dir: $OCI__SYNC_DIR"
echo "sleep duration is $sleep_duration"

bucket=$1
prefix=$2/sync/$JOB_OCID/$JOB_RUN_OCID/

while true; do
  if [[ -d $OCI__SYNC_DIR && -n "$(ls -A $OCI__SYNC_DIR)" ]]; then
    echo "syncing $OCI__SYNC_DIR to $bucket/$prefix"
    $HOME/bin/oci os object sync --auth $auth_method --bucket-name $bucket --prefix $prefix --src-dir $OCI__SYNC_DIR
  else
    echo "nothing to sync in $OCI__SYNC_DIR"
  fi
  if [ "$loop" = false ] ; then
    break
  fi
  sleep $sleep_duration
done
    """
