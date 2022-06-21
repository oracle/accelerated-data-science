#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import os
import fsspec
import subprocess
import yaml
import json
from ads.common.auth import get_signer
from urllib.parse import urlparse
from ads.jobs import Job, DataScienceJobRun
from ads.jobs.builders.runtimes.artifact import Artifact
from ads.opctl.backend.ads_ml_job import MLJobDistributedBackend
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.merger import ConfigMerger
from ads.opctl.utils import OCIAuthContext
from ads.opctl.config.yaml_parsers import YamlSpecParser


def _artifact_file_name(cluster_type, version):
    return f"{cluster_type}_{version}.tar.gz".replace("-", "_")


def initialize_workspace(cluster_type, version):
    base_uri = None
    local_file_name = f".tmp_{_artifact_file_name(cluster_type, version)}".replace(
        "-", "_"
    )
    if os.environ.get("OCI_DISTRIBUTED_DEV_MODE"):
        base_uri = dev_mode_base_uri(cluster_type, version)
    else:
        base_uri = production_mode_base_uri(cluster_type, version)
    artificat_location = os.path.join(
        base_uri, _artifact_file_name(cluster_type, version)
    )
    print(f"Downloading from {artificat_location} to {local_file_name}")
    Artifact.copy_from_uri(uri=artificat_location, to_path=local_file_name)
    os.makedirs("oci_dist_training_artifacts", exist_ok=True)
    try:
        subprocess.call(
            ["tar", "-xvf", local_file_name, "-C", "oci_dist_training_artifacts"]
        )
        print(
            f"Check oci_dist_training_artifacts/{cluster_type.replace('-','_')}/{version}/README.md for build instructions "
        )
    finally:
        if os.path.exists(local_file_name):
            os.remove(local_file_name)


def dev_mode_base_uri(cluster_type, version):
    """
    Temporary method to mock initialization using bucket.
    """
    bucket_name = os.environ.get("OCI_DISTRIBUTED_DEV_MODE_BUCKET")
    namespace = os.environ.get("OCI_DISTRIBUTED_DEV_MODE_NAMESPACE")
    prefix = os.environ.get("OCI_DISTRIBUTED_DEV_MODE_PREFIX")
    base_uri = f"oci://{bucket_name}@{namespace}/{prefix}/"
    return base_uri


def production_mode_base_uri(cluster_type, version):
    return "https://raw.githubusercontent.com/oracle-samples/oci-data-science-ai-samples/master/distributed_training/artifacts/"


def cancel_distributed_run(job_id, cluster_file_name, **kwargs):
    workerJobRunIds = []
    mainjob = None
    if cluster_file_name:
        with open(cluster_file_name) as cf:
            content = yaml.load(cf, yaml.SafeLoader)
            job_id = content["jobId"]
            workerJobRunIds = content.get("workerJobRunIds")
            mainjob = content.get("mainJobRunId")
    with OCIAuthContext(profile=kwargs.get("oci_profile", "DEFAULT")):
        import ads

        ads.set_auth("api_key")
        if workerJobRunIds:
            for id in workerJobRunIds:
                print(f"Cancelling Job Run: {id}")
                try:
                    DataScienceJobRun.from_ocid(id).cancel()
                except Exception as e:
                    print(f"Error cancelling: {e}")
        if mainjob:
            print(f"Cancelling Job Run: {mainjob}")
            DataScienceJobRun.from_ocid(mainjob).cancel()

        else:
            job = Job.from_datascience_job(job_id)
            runs = job.run_list()
            for job_run in runs:
                print(f"Cancelling Job Run: {job_run.id}")
                try:
                    job_run.cancel()
                except Exception as e:
                    print(f"Error cancelling: {e}")
            if not runs:
                print(f"No Job runs found for : {job_id}")


def show_config_info(job_id, work_dir, cluster_file_name, worker_info, **kwargs):
    if cluster_file_name:
        with open(cluster_file_name) as cf:
            content = yaml.load(cf, yaml.SafeLoader)
            job_id = content["jobId"]
            work_dir = content["workDir"]
    config_location = os.path.join(work_dir, job_id)
    scheme = urlparse(
        config_location,
    ).scheme

    oci_auth = (
        get_signer(
            kwargs.get("oci_config") or os.path.expanduser("~/.oci/config"),
            kwargs.get("oci_profile") or "DEFAULT",
        )
        if scheme == "oci"
        else {}
    )
    filesystem = fsspec.filesystem(
        scheme, **oci_auth
    )  # FileSystem class corresponding to the URI scheme.
    main_file = os.path.join(config_location, "MAIN_config.json")
    if filesystem.exists(main_file):
        with fsspec.open(main_file, **oci_auth) as mfile:
            print("Main Info:")
            print(yaml.dump(json.loads(mfile.read())))

        if worker_info:
            files = fsspec.open_files(
                os.path.join(config_location, "WORKER*.json"), **oci_auth
            )
            for fl in files:
                with fsspec.open(
                    f"{'oci://' if oci_auth else ''}{fl.path}", **oci_auth
                ) as f:
                    print(f"Worker Info from {os.path.basename(fl.path)}")
                    print(yaml.dump(json.loads(f.read())))
    else:
        print(
            f"MAIN_config file not found at location {config_location}. If you just started the cluster please wait until the main node is in `inprogress` state"
        )


def generate_docker_compose_yaml(config: dict, output: str, **kwargs):
    """Generates the docker compose YAML from ADS distributed training config.

    Parameters
    ----------
    config : dict
        ADS distributed training config loaded from YAML.
    output : str
        Docker compose YAML output file path.
    """
    oci_config_path = kwargs.get("oci_config")
    if not oci_config_path:
        oci_config_path = "~/.oci"
    p = ConfigProcessor(config).step(ConfigMerger, **kwargs)
    backend = MLJobDistributedBackend(p.config)
    cluster_info = YamlSpecParser.parse_content(config)
    main_jobrun_conf, worker_jobrun_conf_list = backend.prepare_job_config(
        cluster_info=cluster_info
    )

    services = dict()
    envs = dict(
        OCI_IAM_TYPE="api_key",
        SHAPE=cluster_info.infrastructure["spec"]["shapeName"]
    )
    volumes = [
        "~/.oci:/home/datascience/.oci",
        "~/.oci:/root/.oci",
        "./work_dir:/work_dir",
        "./artifacts:/opt/ml"
    ]
    main_env = copy.deepcopy(envs)
    main_env.update(backend.job.runtime.envs)
    main_env.update(main_jobrun_conf.get("envVars", {}))
    services[main_jobrun_conf["name"]] = {
        "image": backend.job.runtime.image,
        "environment": main_env,
        "network_mode": "host",
        "volumes": volumes.copy(),
    }
    for i, worker_jobrun_conf in enumerate(worker_jobrun_conf_list):
        worker_env = copy.deepcopy(envs)
        worker_env.update(backend.job.runtime.envs)
        worker_env.update(worker_jobrun_conf.get("envVars", {}))
        services[backend.generate_worker_name(worker_jobrun_conf, i)] = {
            "image": backend.job.runtime.image,
            "environment": worker_env,
            "network_mode": "host",
            "volumes": volumes.copy(),
        }
    compose = dict(services=services)
    if output:
        output = os.path.abspath(os.path.expanduser(output))
        with open(output, "w", encoding="utf-8") as f:
            yaml.dump(compose, f)
        print(f"Docker compose file saved to {output}")
    else:
        print("=" * 80)
        print("docker-compose.yml")
        print("=" * 80)
        print(yaml.dump(compose))
        print("=" * 80)
    print(
        "Please add any port mappings for communication, for example:\n"
        "ports:\n"
        '- "30000:29400"\n'
    )
