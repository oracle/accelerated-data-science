#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


OCI__RUNTIME_TYPE = "OCI__RUNTIME_TYPE"
OCI__RUNTIME_URI = "OCI__RUNTIME_URI"
OCI__RUNTIME_PYTHON_PATH = "OCI__RUNTIME_PYTHON_PATH"
OCI__RUNTIME_GIT_BRANCH = "OCI__RUNTIME_GIT_BRANCH"
OCI__RUNTIME_GIT_COMMIT = "OCI__RUNTIME_GIT_COMMIT"
OCI__RUNTIME_GIT_SECRET_ID = "OCI__RUNTIME_GIT_SECRET_ID"
OCI__CODE_DIR = "OCI__CODE_DIR"


class ClusterConfigToJobSpecConverter:
    def __init__(self, cluster_info):
        self.cluster_info = cluster_info

    def job_def_info(self):
        job = {}
        job["infrastructure"] = self.cluster_info.infrastructure
        cluster = self.cluster_info.cluster
        job["image"] = cluster.image or cluster.main.image or cluster.worker.image
        job["name"] = cluster.name or self.cluster_info.infrastructure.get(
            "displayName"
        )
        job["envVars"] = cluster.config.envVars
        job["envVars"]["OCI__WORK_DIR"] = cluster.work_dir
        job["envVars"]["OCI__EPHEMERAL"] = cluster.ephemeral
        job["envVars"]["OCI__CLUSTER_TYPE"] = cluster.type.upper()
        job["envVars"]["OCI__WORKER_COUNT"] = (
            cluster.worker.replicas if cluster.worker is not None else 0
        )
        if cluster.ps is not None:
            job["envVars"]["OCI__PS_COUNT"] = cluster.ps.replicas

        job["envVars"]["OCI__START_ARGS"] = cluster.config.cmd_args.strip()
        job["envVars"]["OCI__ENTRY_SCRIPT"] = self.cluster_info.runtime.entry_point

        if self.cluster_info.runtime.type:
            optional_runtime_fields = [
                {"attr": "type", "env": OCI__RUNTIME_TYPE},
                {"attr": "uri", "env": OCI__RUNTIME_URI},
                {"attr": "branch", "env": OCI__RUNTIME_GIT_BRANCH},
                {"attr": "commit", "env": OCI__RUNTIME_GIT_COMMIT},
                {"attr": "git_secret_id", "env": OCI__RUNTIME_GIT_SECRET_ID},
                {"attr": "code_dir", "env": OCI__CODE_DIR},
                {"attr": "python_path", "env": OCI__RUNTIME_PYTHON_PATH},
            ]
            for field in optional_runtime_fields:
                val = getattr(self.cluster_info.runtime, field["attr"], None)
                if val:
                    job["envVars"][field["env"]] = val

        runtime_args = self.cluster_info.runtime.args
        if isinstance(self.cluster_info.runtime.args, list):
            runtime_args = " ".join([str(v) for v in self.cluster_info.runtime.args])
        if runtime_args:
            job["envVars"]["OCI__ENTRY_SCRIPT_ARGS"] = runtime_args
        if self.cluster_info.runtime.kwargs:
            job["envVars"][
                "OCI__ENTRY_SCRIPT_KWARGS"
            ] = self.cluster_info.runtime.kwargs
        job["envVars"].update(self.cluster_info.runtime.envVars)
        job["envVars"] = {k: str(job["envVars"][k]) for k in job["envVars"]}

        if self.cluster_info.cluster.certificate:
            job["envVars"][
                "OCI__CERTIFICATE_OCID"
            ] = self.cluster_info.cluster.certificate.cert_ocid
            job["envVars"][
                "OCI__CERTIFICATE_KEY_DOWNLOAD_LOCATION"
            ] = self.cluster_info.cluster.certificate.key_download_location
            job["envVars"][
                "OCI__CERTIFICATE_DOWNLOAD_LOCATION"
            ] = self.cluster_info.cluster.certificate.cert_download_location
            job["envVars"][
                "OCI__CERTIFICATE_AUTHORITY_OCID"
            ] = self.cluster_info.cluster.certificate.ca_ocid
            job["envVars"][
                "OCI__CA_DOWNLOAD_LOCATION"
            ] = self.cluster_info.cluster.certificate.ca_download_location

        return job

    def job_run_info(self, jobType):
        jobrun = {}
        jobTypeConfig = getattr(self.cluster_info.cluster, jobType)
        if jobTypeConfig is not None:
            jobrun["name"] = jobTypeConfig.name or jobType
            jobrun["envVars"] = jobTypeConfig.config.envVars
            if jobTypeConfig.config.cmd_args:
                jobrun["envVars"][
                    "OCI__START_ARGS"
                ] = jobTypeConfig.config.cmd_args.strip()

            jobrun["envVars"]["OCI__MODE"] = jobType.upper()
            jobrun["envVars"] = {
                k: str(jobrun["envVars"][k]) for k in jobrun["envVars"]
            }
        return jobrun
