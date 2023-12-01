#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
import re
import subprocess
from configparser import ConfigParser
from urllib.parse import urlparse

import fsspec
import yaml

from ads.common.auth import AuthContext, AuthType, create_signer
from ads.common.decorator.runtime_dependency import (
    OptionalDependency,
    runtime_dependency,
)
from ads.jobs import Job
from ads.jobs.builders.runtimes.artifact import Artifact
from ads.opctl.utils import publish_image as publish_image_cmd
from ads.opctl.utils import run_command

ini_file = "config.ini"


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
            f"Check oci_dist_training_artifacts/{cluster_type.replace('-', '_')}/{version}/README.md for build instructions "
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
    if cluster_file_name:
        with open(cluster_file_name) as cf:
            content = yaml.load(cf, yaml.SafeLoader)
            job_id = content["jobId"]

    with AuthContext():
        import ads

        ads.set_auth(auth="api_key", profile=kwargs.get("oci_profile", "DEFAULT"))

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
        create_signer(
            AuthType.API_KEY,
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


@runtime_dependency(module="docker", install_from=OptionalDependency.OPCTL)
def verify_image(img_name):
    """
    Verify if the input image exists in OCI registry

    Parameters
    ----------
    img_name
        Name of the docker image

    Returns
        None
    -------

    """
    import docker

    client = docker.from_env()
    try:
        client.images.get_registry_data(img_name)
        return 1
    except Exception:
        return 0


def update_ini(tag, registry, dockerfile, source_folder, config, nobuild):
    """
    Stores and updates input args in config.ini in Local

    Parameters
    ----------
    tag
        tag of image
    registry
        registry to push to
    dockerfile
        relative path to Dockerfile
    source_folder
        source scripts folder that will be mounted during a local run
    config
        Job config
    nobuild
        flag for building the docker

    Returns
        dict of config.ini file
    -------

    """
    ini = ConfigParser(allow_no_value=True)
    if config is None:
        img_name = "@test"
    else:
        img_name = (
            config.get("spec", {}).get("cluster", {}).get("spec", {}).get("image")
        )

    if tag is not None:
        tag1 = tag
    else:
        if img_name.startswith("@"):
            tag1 = tag
        else:
            if len(img_name.rsplit(":", 1)) == 1:
                tag1 = "latest"
            else:
                tag1 = img_name.rsplit(":", 1)[1]
    if registry is not None:
        registry1 = registry
    else:
        if img_name.startswith("@"):
            registry1 = registry
        else:
            registry1 = img_name.rsplit(":", 1)[0]

    if os.path.isfile(ini_file):
        ini.read(ini_file)
        if tag1 is not None:
            ini.set("main", "tag", tag1)
        if registry1 is not None:
            ini.set("main", "registry", registry1)
        if dockerfile is not None:
            ini.set("main", "dockerfile", dockerfile)
        if source_folder is not None:
            ini.set("main", "source_folder", source_folder)
    else:
        ini.add_section("main")
        if tag1 is not None:
            ini.set("main", "tag", tag1)
        else:
            raise ValueError("tag arg is missing")
        if registry1 is not None:
            ini.set("main", "registry", registry1)
        else:
            raise ValueError("registry arg is missing")
        if dockerfile is not None:
            ini.set("main", "dockerfile", dockerfile)
        else:
            if nobuild:
                ini.set("main", "dockerfile", "DUMMY_PATH")
            else:
                raise ValueError("dockerfile arg is missing")
        if source_folder is not None:
            ini.set("main", "source_folder", source_folder)
        else:
            ini.set("main", "source_folder", ".")
    ini.set("main", "; mount oci keys for local testing", None)
    if ini.has_option("main", "oci_key_mnt"):
        pass
    else:
        ini.set("main", "oci_key_mnt", "~/.oci:/home/oci_dist_training/.oci")
    if ini.get("main", "dockerfile") == "DUMMY_PATH" and not nobuild:
        raise ValueError("dockerfile arg is missing")
    if os.path.exists(ini_file):
        os.remove(ini_file)
    with open(ini_file, "w") as f:
        ini.write(f)
    return ini


def load_ini():
    """
    Loads config.ini from local in dict
    Returns
        None
    -------

    """
    ini = ConfigParser(allow_no_value=True)
    if os.path.isfile(ini_file):
        ini.read(ini_file)
        return ini
    else:
        raise RuntimeError(f"ini_file file {ini_file} not found !")


def increment_tag_in_ini(ini):
    """
    increments tag of image and update latest in config.ini file

    Parameters
    ----------
    ini
        config.ini file dictionary
    Returns
        updated config.ini file
    -------

    """
    ini = increment_tag(ini)
    refresh_ini(ini, "config.ini")
    return ini


def increment_tag(ini):
    """
    Increments the tag of the image

    Parameters
    ----------
    ini
        config.ini file dictionary

    Returns
        updated ini dict
    -------

    """
    tag_name = ini.get("main", "tag")
    m = re.search(r"_v_\d+$", tag_name)
    if m is not None:
        updated_tag = re.sub(
            r"[0-9]+$",
            lambda x: f"{str(int(x.group()) + 1).zfill(len(x.group()))}",
            tag_name,
        )
    else:
        updated_tag = f"{tag_name}_v_1"
    ini.set("main", "tag", updated_tag)
    return ini


def refresh_ini(ini, ini_file):
    """
    write updated config.ini in local

    Parameters
    ----------
    ini
        config.ini file dictionary
    ini_file
        Name of the config file

    Returns
    -------

    """
    if os.path.exists(ini_file):
        os.remove(ini_file)
    with open(ini_file, "w") as f:
        ini.write(f)
    return ini


def docker_build_cmd(ini):
    """
    Builds the docker image

    Parameters
    ----------
    ini
        config.ini file dictionary

    Returns
        None
    -------

    """
    cmd = get_cmd(ini)
    return run_cmd(cmd)


def get_cmd(ini):
    """
    Docker build command

    Parameters
    ----------
    ini
        config.ini file dictionary
    Returns
        Docker build command
    -------

    """
    command = [
        "docker",
        "build",
        "--build-arg",
        "CODE_DIR=" + ini.get("main", "source_folder"),
        "-t",
        ini.get("main", "registry") + ":" + ini.get("main", "tag"),
        "-f",
        ini.get("main", "dockerfile"),
    ]
    if os.environ.get("no_proxy"):
        command += ["--build-arg", f"no_proxy={os.environ['no_proxy']}"]
    if os.environ.get("http_proxy"):
        command += ["--build-arg", f"http_proxy={os.environ['http_proxy']}"]
    if os.environ.get("https_proxy"):
        command += ["--build-arg", f"https_proxy={os.environ['https_proxy']}"]
    command += ["."]
    return command


def run_cmd(cmd):
    """
    Run cli command
    Parameters
    ----------
    cmd
        input cli command

    Returns
        None
    -------

    """
    proc = run_command(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Docker build failed.")
    return 1


def horovod_cmd(code_mount, oci_key_mount, config):
    """
    CLI command to run horovod distributed framework on local

    Parameters
    ----------
    code_mount
        source_folder to be mounted
    oci_key_mount
        oci keys to be mounted
    config
        Job config

    Returns
        List: CLI command
    -------

    """
    command = [
        "docker",
        "run",
        "-v",
        code_mount,
        "-v",
        oci_key_mount,
        "--env",
        "OCI_IAM_TYPE=api_key",
        "--rm",
        "--entrypoint",
        "/miniconda/envs/env/bin/horovodrun",
        config["spec"]["cluster"]["spec"]["image"],
        "--gloo",
        "-np",
        "2",
        "-H",
        "localhost:2",
        "/miniconda/envs/env/bin/python",
        config["spec"]["runtime"]["spec"]["entryPoint"],
    ]
    return command


def pytorch_cmd(code_mount, oci_key_mount, config):
    """
    CLI command to run Pytorch distributed framework on local

    Parameters
    ----------
    code_mount
        source_folder to be mounted
    oci_key_mount
        oci keys to be mounted
    config
        Job config

    Returns
        List: CLI command
    -------

    """
    command = [
        "docker",
        "run",
        "-v",
        code_mount,
        "-v",
        oci_key_mount,
        "--env",
        "OCI_IAM_TYPE=api_key",
        "--env",
        "WORLD_SIZE=1",
        "--env",
        "RANK=0",
        "--env",
        "LOCAL_RANK=0",
        "--rm",
        "--entrypoint",
        "/opt/conda/bin/python",
        config["spec"]["cluster"]["spec"]["image"],
        config["spec"]["runtime"]["spec"]["entryPoint"],
    ]
    return command


def dask_cmd(code_mount, oci_key_mount, config):
    """
    CLI command to run Dask distributed framework on local

    Parameters
    ----------
    code_mount
        source_folder to be mounted
    oci_key_mount
        oci keys to be mounted
    config
        Job config

    Returns
        List: CLI command
    -------

    """
    command = [
        "docker",
        "run",
        "-v",
        code_mount,
        "-v",
        oci_key_mount,
        "--env",
        "OCI_IAM_TYPE=api_key",
        "--env",
        "SCHEDULER_IP=tcp://127.0.0.1",
        "--rm",
        "--entrypoint",
        "/bin/sh",
        config["spec"]["cluster"]["spec"]["image"],
        "-c",
        "(nohup dask-scheduler >scheduler.log &) && (nohup dask-worker localhost:8786 >worker.log &) && "
        "/miniconda/envs/daskenv/bin/python "
        + config["spec"]["runtime"]["spec"]["entryPoint"],
    ]
    return command


def tensorflow_cmd(code_mount, oci_key_mount, config):
    """
    CLI command to run Tensorflow distributed framework on local

    Parameters
    ----------
    code_mount
        source_folder to be mounted
    oci_key_mount
        oci keys to be mounted
    config
        Job config

    Returns
        List: CLI command
    -------

    """
    if "ps" in config["spec"]["cluster"]["spec"]:
        # this is for the parameter server strategy.
        command = [
            "docker",
            "run",
            "-v",
            code_mount,
            "-v",
            oci_key_mount,
            "--env",
            "OCI_IAM_TYPE=api_key",
            "--rm",
            "--entrypoint",
            "/bin/sh",
            config["spec"]["cluster"]["spec"]["image"],
            "/etc/datascience/local_test.sh",
            config["spec"]["runtime"]["spec"]["entryPoint"],
        ]

    else:
        # this is for the mirror and multiworkermirror strategy.
        command = [
            "docker",
            "run",
            "-v",
            code_mount,
            "-v",
            oci_key_mount,
            "--env",
            "OCI_IAM_TYPE=api_key",
            "--env",
            'TF_CONFIG={"cluster": {"worker": ["localhost:12345"]}, "task": {"type": "worker", "index": 0}}',
            "--rm",
            "--entrypoint",
            "/miniconda/bin/python",
            config["spec"]["cluster"]["spec"]["image"],
            config["spec"]["runtime"]["spec"]["entryPoint"],
        ]
    return command


def local_run(config, ini):
    """

    Local run distributed framework on local based on input args

    Parameters
    ----------
    ini
        config.ini file dictionary
    config
        Job run config

    Returns
        None
    -------

    """
    cwd = os.path.join(os.getcwd(), ini.get("main", "source_folder"))
    code_mount = os.path.join(cwd, ":/code/")
    oci_key_mount = (
        os.path.expanduser(ini.get("main", "oci_key_mnt").split(":")[0])
        + ":"
        + ini.get("main", "oci_key_mnt").split(":")[1]
    )
    if config["spec"]["cluster"]["kind"].lower() == "horovod":
        command = horovod_cmd(code_mount, oci_key_mount, config)
    elif config["spec"]["cluster"]["kind"].lower() == "pytorch":
        command = pytorch_cmd(code_mount, oci_key_mount, config)
    elif config["spec"]["cluster"]["kind"].lower() == "dask":
        command = dask_cmd(code_mount, oci_key_mount, config)
    elif config["spec"]["cluster"]["kind"].lower() == "tensorflow":
        command = tensorflow_cmd(code_mount, oci_key_mount, config)
    else:
        raise RuntimeError(f"Framework not supported")
    try:
        command += [str(arg) for arg in config["spec"]["runtime"]["spec"]["args"]]
    except KeyError:
        pass
    print("Running: ", " ".join(command))
    proc = run_command(command)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to run local")
    return 1


def update_image(config, ini):
    """
    Update Image name in job config with latest build image

    Parameters
    ----------
    config
        Job run config
    ini
        config.ini file dictionary

    Returns
        Updated Job run config
    -------

    """
    config["spec"]["cluster"]["spec"]["image"] = (
        ini.get("main", "registry") + ":" + ini.get("main", "tag")
    )
    return config


def verify_and_publish_image(nopush, config):
    """
    verify existence of docker image in OCI registry and publish it
    Parameters
    ----------
    nopush
        Doesn't push image to OCI registry
    config
        Job run config

    Returns
        None
    -------

    """
    if not nopush:
        publish_image_cmd(config["spec"]["cluster"]["spec"]["image"])
    else:
        if not verify_image(config["spec"]["cluster"]["spec"]["image"]):
            print(
                "\u26A0 Image: "
                + config["spec"]["cluster"]["spec"]["image"]
                + " does not exist in registry"
            )
            print("In order to push the image to registry enter Y else N ")
            inp = input("[Y/N]\n")
            if inp == "Y":
                print("\u2705 pushing image to registry")
                publish_image_cmd(config["spec"]["cluster"]["spec"]["image"])
            else:
                raise RuntimeError(
                    "Stopping the execution as image doesn't exist in OCI registry"
                )
    return 1


def update_config_image(config):
    """
    updates image name in config

    Parameters
    ----------
    config
        Job config

    Returns
        updated config dict
    -------

    """
    ini = ConfigParser(allow_no_value=True)
    img_name = config.get("spec", {}).get("cluster", {}).get("spec", {}).get("image")
    if img_name.startswith("@"):
        if os.path.isfile(ini_file):
            ini.read(ini_file)
            return update_image(config, ini)
        else:
            raise ValueError(
                f"Invalid image name: {img_name} and also not able to locate config.ini file. "
                f"Please update image name in Job config."
            )
    else:
        return config
