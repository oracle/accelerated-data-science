#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import datetime
import os
import shutil
import subprocess
import shlex
import json
import glob
from typing import Dict

import click
import yaml

from datetime import datetime
import ocifs

from ads.common.oci_client import OCIClientFactory
from ads.common.auth import create_signer
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.config import NO_CONTAINER

from ads.opctl.constants import (
    ML_JOB_GPU_IMAGE,
    ML_JOB_IMAGE,
    DEFAULT_IMAGE_HOME_DIR,
    DEFAULT_IMAGE_CONDA_DIR,
    DEFAULT_NOTEBOOK_SESSION_SPARK_CONF_DIR,
)
from ads.opctl import logger
from ads.opctl.config.utils import read_from_ini
from ads.opctl.utils import (
    parse_conda_uri,
    run_container,
    get_docker_client,
    is_in_notebook_session,
    run_command,
)
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.merger import ConfigMerger
from ads.opctl.conda.multipart_uploader import MultiPartUploader


def _fetch_manifest_template() -> Dict:
    manifest_template_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "manifest_template.yaml"
    )
    with open(manifest_template_file) as manifest_file:
        manifest_template = yaml.safe_load(manifest_file)
    return manifest_template


@runtime_dependency(module="docker", install_from=OptionalDependency.OPCTL)
def _check_job_image_exists(gpu: bool) -> None:
    if gpu:
        image = ML_JOB_GPU_IMAGE
    else:
        image = ML_JOB_IMAGE
    try:
        client = get_docker_client()
        client.api.inspect_image(image)
    except docker.errors.ImageNotFound:
        if gpu:
            cmd = "`ads opctl build-image -g job-local`"
        else:
            cmd = "`ads opctl build-image job-local`"
        raise RuntimeError(
            f"Please run {cmd} to build a local image for Jobs development first."
        )


def _get_name(name: str, env_file: str) -> str:
    if not name and env_file:
        with open(env_file) as f:
            name = yaml.safe_load(f.read()).get("name", None)
    if not name:
        raise ValueError(
            "Either specify environment name in environment yaml or with `--name`."
        )
    return name


def create(**kwargs) -> str:
    p = ConfigProcessor().step(ConfigMerger, **kwargs)
    exec_config = p.config["execution"]
    name = _get_name(exec_config.get("name"), exec_config.get("environment_file"))
    _create(
        name=name,
        version=exec_config["version"],
        env_file=exec_config["environment_file"],
        conda_pack_folder=exec_config["conda_pack_folder"],
        gpu=exec_config.get("gpu", False),
        overwrite=exec_config.get("overwrite", False),
    )


def _create(
    name: str,
    version: str,
    env_file: str,
    conda_pack_folder: str,
    gpu: bool,
    overwrite: bool,
) -> str:
    """Create a conda pack given an environment yaml file under conda pack folder specified.

    Parameters
    ----------
    slug : str
        slug of the conda pack
    env_file : str
        path to conda environment yaml
    conda_pack_folder : str
        path to conda pack folder
    gpu : bool
        whether to build against GPU image
    overwrite : bool
        whether to overwrite existing pack of the same slug

    Raises
    ------
    FileNotFoundError
        Environment file not found
    RuntimeError
        Environment file may be inproperly removed
    RuntimeError
        Creating pack failed

    Return
    ------
    str:
        slug of the environment
    """

    if not os.path.exists(env_file):
        raise FileNotFoundError(f"Environment file {env_file} is not found.")

    slug = f"{name}_v{version}".replace(" ", "").replace(".", "_").lower()
    pack_folder_path = os.path.join(
        os.path.abspath(os.path.expanduser(conda_pack_folder)), slug
    )

    if os.path.exists(pack_folder_path):
        overwrite = overwrite or click.confirm(
            f"Conda pack with slug {slug} already exists. Do you wish to overwrite?"
        )
        if overwrite:
            if (
                os.path.commonpath(
                    [pack_folder_path, os.path.abspath(os.path.expanduser(env_file))]
                )
                == pack_folder_path
            ):
                raise RuntimeError(
                    f"Environment file {os.path.abspath(os.path.expanduser(env_file))} is in {pack_folder_path}. Overwriting this folder will remove the file."
                )
            shutil.rmtree(pack_folder_path)
        else:
            return

    os.makedirs(pack_folder_path, exist_ok=True)

    manifest = _fetch_manifest_template()
    manifest["manifest"]["name"] = name
    manifest["manifest"]["slug"] = slug
    manifest["manifest"]["type"] = "published"
    manifest["manifest"]["version"] = version
    manifest["manifest"]["arch_type"] = "GPU" if gpu else "CPU"

    manifest["manifest"]["create_date"] = datetime.utcnow().strftime(
        "%a, %b %d, %Y, %H:%M:%S %Z UTC"
    )
    manifest["manifest"]["manifest_version"] = "1.0"

    logger.info(f"Creating conda environment {slug}")
    if is_in_notebook_session() or NO_CONTAINER:
        command = f"conda env create --prefix {pack_folder_path} --file {os.path.abspath(os.path.expanduser(env_file))}"
        run_command(command, shell=True)
    else:
        _check_job_image_exists(gpu)
        docker_pack_folder_path = os.path.join(DEFAULT_IMAGE_HOME_DIR, slug)
        docker_env_file_path = os.path.join(
            DEFAULT_IMAGE_HOME_DIR, os.path.basename(env_file)
        )

        create_command = f"conda env create --prefix {docker_pack_folder_path} --file {docker_env_file_path}"

        volumes = {
            pack_folder_path: {"bind": docker_pack_folder_path},
            os.path.abspath(os.path.expanduser(env_file)): {
                "bind": docker_env_file_path
            },
        }
        if gpu:
            image = ML_JOB_GPU_IMAGE
        else:
            image = ML_JOB_IMAGE
        try:
            run_container(
                image=image, bind_volumes=volumes, env_vars={}, command=create_command
            )
        except Exception:
            if os.path.exists(pack_folder_path):
                shutil.rmtree(pack_folder_path)
            raise RuntimeError(f"Could not create environment {slug}.")

    conda_dep = None
    with open(env_file) as mfile:
        conda_dep = yaml.safe_load(mfile.read())
    conda_dep["manifest"] = manifest["manifest"]
    with open(f"{os.path.join(pack_folder_path, slug)}_manifest.yaml", "w") as mfile:
        yaml.safe_dump(conda_dep, mfile)

    logger.info(f"Environment `{slug}` setup complete.")
    print(f"Pack {slug} created under {pack_folder_path}.")
    return slug


def _fetch_pack_path(
    slug: str,
    conda_pack_os_prefix: str,
    oci_config: str,
    oci_profile: str,
    auth_type: str,
) -> str:
    oci_auth = create_signer(auth_type, oci_config, oci_profile)
    fs = ocifs.OCIFileSystem(**oci_auth)
    fnames = fs.ls(conda_pack_os_prefix, detail=True, refresh=True)
    for f in fnames:
        if os.path.basename(f["name"]) == slug:
            return f"oci://{f['name']}"
        elif f["type"] == "directory":
            path = _fetch_pack_path(
                slug, f"oci://{f['name']}", oci_config, oci_profile, auth_type
            )
            if path:
                return path
    return None


def install(**kwargs) -> None:
    p = ConfigProcessor().step(ConfigMerger, **kwargs)
    exec_config = p.config["execution"]
    if exec_config.get("slug") and exec_config.get("conda_pack_os_prefix"):
        conda_uri = _fetch_pack_path(
            exec_config["slug"],
            exec_config["conda_pack_os_prefix"],
            exec_config.get("oci_config"),
            exec_config.get("oci_profile"),
            exec_config.get("auth"),
        )
        if not conda_uri:
            raise FileNotFoundError(
                f"{exec_config['slug']} not found under {exec_config['conda_pack_os_prefix']}"
            )
    elif exec_config.get("conda_uri"):
        conda_uri = exec_config["conda_uri"]
    else:
        raise ValueError(
            "Either `--conda-uri`, or `--slug` and `--conda-pack-os-prefix` has to be specified."
        )
    _install(
        conda_uri,
        os.path.abspath(os.path.expanduser(exec_config["conda_pack_folder"])),
        exec_config.get("oci_config"),
        exec_config.get("oci_profile"),
        overwrite=exec_config.get("overwrite", False),
        debug=kwargs.get("debug", False),
        auth_type=exec_config.get("auth"),
    )


def _install(
    conda_uri: str,
    conda_pack_folder: str,
    oci_config: str = None,
    oci_profile: str = None,
    overwrite: bool = False,
    debug: bool = False,
    auth_type: str = None,
) -> None:
    """
    Install conda pack.

    Parameters
    ----------
    conda_uri: str
        OCI object storage uri to conda pack
    conda_pack_folder: str
        local folder to save conda packs
    oci_config: str
        path to OCI config file
    oci_profile: str
        OCI config profile
    overwrite: bool
        whether to overwrite existing pack
    debug: bool
        whether to turn on debug mode
    auth_type : str
        authentication method
    Returns
    -------
    None
    """
    ns, bucket, path, slug = parse_conda_uri(conda_uri)
    if bucket == "service-conda-packs":
        raise ValueError(
            "Download service conda pack is not allowed. Only custom conda pack can be downloaded to local machine. You need to publish it to your own bucket first."
        )
    os.makedirs(conda_pack_folder, exist_ok=True)
    pack_path = os.path.join(os.path.expanduser(conda_pack_folder), slug + ".tar.gz")
    pack_folder_path = os.path.join(os.path.expanduser(conda_pack_folder), slug)

    if not (is_in_notebook_session() or NO_CONTAINER):
        _check_job_image_exists(gpu=False)

    while os.path.exists(pack_folder_path):
        if overwrite:
            break
        else:
            ans = click.prompt(
                f"conda pack with slug {slug} already exists. Enter a new name or 'o' for overwrite:",
                default="o",
            )
            if ans == "o":
                overwrite = True
                break
            else:
                slug = ans
                pack_path = os.path.join(
                    os.path.expanduser(conda_pack_folder), slug + ".tar.gz"
                )
                pack_folder_path = os.path.join(
                    os.path.expanduser(conda_pack_folder), slug
                )

    if oci_config is None or oci_profile is None:
        download_command = [
            "oci",
            "os",
            "object",
            "get",
            "--name",
            f"{path}",
            "-bn",
            bucket,
            "-ns",
            ns,
            "--file",
            pack_path,
            "--auth",
            auth_type,
        ]
    else:
        oci_config = os.path.abspath(os.path.expanduser(oci_config))
        parser = read_from_ini(os.path.expanduser(oci_config))
        if oci_profile not in parser:
            raise ValueError(f"PROFILE {oci_profile} not found in {oci_config}.")
        download_command = [
            "oci",
            "os",
            "object",
            "get",
            "--name",
            f"{path}",
            "-bn",
            bucket,
            "-ns",
            ns,
            "--file",
            pack_path,
            "--profile",
            oci_profile,
            "--config-file",
            oci_config,
        ]
    if debug:
        download_command.append("-d")
        print(" ".join([shlex.quote(c) for c in download_command]))
    process = subprocess.Popen(download_command)
    process.communicate()
    if process.returncode != 0:
        if os.path.exists(pack_path):
            os.remove(pack_path)
        raise RuntimeError(
            f"Downloading pack failed with return code {process.returncode}. Please double check the path and make sure you have access."
        )
    else:
        print(f"Download {conda_uri} completed")

    print(f"Start unpacking {pack_path}")
    os.makedirs(pack_folder_path, exist_ok=True)

    process = subprocess.Popen(["tar", "-xf", pack_path, "-C", pack_folder_path])
    process.communicate()
    if process.returncode != 0:
        shutil.rmtree(pack_folder_path)
        raise Exception(
            f"Error extracting {pack_path} to {pack_folder_path}. Please try again."
        )

    # Run the conda-unpack script to clean up prefixes
    # This will fix problems related to repeated "placehold" paths.
    # See https://conda.github.io/conda-pack/#commandline-usage
    unpack_script = os.path.join(pack_folder_path, "bin", "conda-unpack")
    if os.path.exists(unpack_script):
        if is_in_notebook_session() or NO_CONTAINER:
            run_command(unpack_script)
        else:
            volumes = {
                pack_folder_path: {"bind": os.path.join(DEFAULT_IMAGE_HOME_DIR, slug)}
            }
            try:
                run_container(
                    image=ML_JOB_IMAGE,
                    bind_volumes=volumes,
                    env_vars={},
                    command=os.path.join(
                        DEFAULT_IMAGE_HOME_DIR, slug, "bin/conda-unpack"
                    ),
                )
            except Exception:
                raise RuntimeError(f"Error unpacking environment {slug}.")
        if os.path.exists(os.path.join(pack_folder_path, "spark-defaults.conf")):
            if is_in_notebook_session() or NO_CONTAINER:
                if not os.path.exists(DEFAULT_NOTEBOOK_SESSION_SPARK_CONF_DIR):
                    os.makedirs(DEFAULT_NOTEBOOK_SESSION_SPARK_CONF_DIR)
                    shutil.copy(
                        os.path.join(pack_folder_path, "spark-defaults.conf"),
                        DEFAULT_NOTEBOOK_SESSION_SPARK_CONF_DIR,
                    )
                    for file in os.listdir(pack_folder_path):
                        if os.path.splitext(file)[1] == ".jar":
                            shutil.copy(
                                os.path.join(pack_folder_path, file),
                                DEFAULT_NOTEBOOK_SESSION_SPARK_CONF_DIR,
                            )
            else:
                with open(os.path.join(pack_folder_path, "spark-defaults.conf")) as f:
                    confs = f.read()
                confs = confs.replace(
                    DEFAULT_NOTEBOOK_SESSION_SPARK_CONF_DIR,
                    os.path.join(DEFAULT_IMAGE_CONDA_DIR, slug),
                )
                with open(
                    os.path.join(pack_folder_path, "spark-defaults.conf"), "w"
                ) as f:
                    f.write(confs)

    os.remove(pack_path)
    manifest_path = glob.glob(os.path.join(pack_folder_path, "*_manifest.yaml"))[0]
    with open(manifest_path) as f:
        env = yaml.safe_load(f.read())
    env["manifest"]["pack_uri"] = conda_uri
    with open(manifest_path, "w") as f:
        yaml.safe_dump(env, f)
    print(f"{slug} set up at {pack_folder_path}.")


def publish(**kwargs) -> None:
    p = ConfigProcessor().step(ConfigMerger, **kwargs)
    exec_config = p.config["execution"]
    if exec_config.get("environment_file", None):
        name = _get_name(exec_config.get("name"), exec_config.get("environment_file"))
        slug = _create(
            name=name,
            version=exec_config["version"],
            env_file=exec_config["environment_file"],
            conda_pack_folder=exec_config["conda_pack_folder"],
            gpu=exec_config.get("gpu", False),
            overwrite=exec_config["overwrite"],
        )
    else:
        slug = exec_config.get("slug")
    if not slug:
        raise ValueError("Please specify slug of the conda pack via `--slug`.")
    if not exec_config.get("conda_pack_os_prefix"):
        raise ValueError(
            "Please specify object storage path to save conda pack either via `ads opctl configure` or `--conda-pack-os-prefix`."
        )
    _publish(
        conda_slug=slug,
        conda_uri_prefix=exec_config["conda_pack_os_prefix"],
        conda_pack_folder=exec_config["conda_pack_folder"],
        oci_config=exec_config.get("oci_config"),
        oci_profile=exec_config.get("oci_profile"),
        overwrite=exec_config["overwrite"],
        auth_type=exec_config["auth"],
    )


def _publish(
    conda_slug: str,
    conda_uri_prefix: str,
    conda_pack_folder: str,
    oci_config: str,
    oci_profile: str,
    overwrite: bool,
    auth_type: str,
) -> None:
    """Publish a local conda pack to object storage location

    Parameters
    ----------
    conda_slug : str
        slug of conda pack
    conda_uri_prefix : str
        object storage prefix to save conda pack
    conda_pack_folder : str
        path to local conda folder
    oci_config : str
        oci config file path
    oci_profile : str
        oci config profile
    overwrite : bool
        whether to overwrite existing pack of the same slug
    auth_type : str
        authentication method

    Raises
    ------
    FileNotFoundError
        local conda pack folder not found
    FileNotFoundError
        manifest file not found inside conda pack
    RuntimeError
        IP packages found inside conda pack
    RuntimeError
        Packing conda failed
    RuntimeError
        Pack file not found
    """
    ns, bucket, prefix, _ = parse_conda_uri(conda_uri_prefix)

    # ======= check if conda pack and manifest exists ==============
    pack_folder_path = os.path.abspath(
        os.path.expanduser(os.path.join(conda_pack_folder, conda_slug))
    )
    if not os.path.exists(pack_folder_path):
        raise FileNotFoundError(
            f"Could not find environment {conda_slug} in {conda_pack_folder}."
        )
    paths = glob.glob(os.path.join(pack_folder_path, "*_manifest.yaml"))
    if len(paths) != 1:
        raise FileNotFoundError(
            "Could not locate manifest file in the provided environment."
        )
    else:
        manifest_location = paths[0]

    # ===== check if pack contains IP packages =========
    print(f"Loading environment information from {manifest_location}.")
    with open(manifest_location) as mlf:
        env = yaml.safe_load(mlf.read())
    if "IP" in env["manifest"] and env["manifest"]["IP"].lower() == "y":
        raise RuntimeError("This environment has IP restricted packages.")

    oci_auth = create_signer(auth_type, oci_config, oci_profile)
    client = OCIClientFactory(**oci_auth).object_storage
    publish_slug = conda_slug
    if not overwrite:
        existing_packs = client.list_objects(ns, bucket, prefix=prefix).data.objects
        pack_names = [os.path.basename(pack.name) for pack in existing_packs]
        while publish_slug in pack_names:
            ans = click.prompt(
                f"{conda_slug} exists in {conda_uri_prefix}. Enter a new name or overwrite (o)",
                default="o",
            )
            if ans == "o":
                break
            else:
                publish_slug = "_".join(ans.lower().split(" "))

    pack_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pack.py")
    if is_in_notebook_session() or NO_CONTAINER:
        command = f"python {pack_script} {pack_folder_path}"
        run_command(command, shell=True)
    else:
        volumes = {
            pack_folder_path: {
                "bind": os.path.join(DEFAULT_IMAGE_HOME_DIR, conda_slug)
            },
            pack_script: {"bind": os.path.join(DEFAULT_IMAGE_HOME_DIR, "pack.py")},
        }
        command = f"python {os.path.join(DEFAULT_IMAGE_HOME_DIR, 'pack.py')} {os.path.join(DEFAULT_IMAGE_HOME_DIR, conda_slug)}"
        gpu = env["manifest"]["arch_type"] == "GPU"
        _check_job_image_exists(gpu)
        if gpu:
            image = ML_JOB_GPU_IMAGE
        else:
            image = ML_JOB_IMAGE
        try:
            run_container(
                image=image, bind_volumes=volumes, env_vars={}, command=command
            )
        except Exception:
            raise RuntimeError(f"Could not pack environment {conda_slug}.")

    pack_file = os.path.join(pack_folder_path, f"{conda_slug}.tar.gz")
    if not os.path.exists(pack_file):
        raise RuntimeError(f"Pack {pack_file} was not created.")
    pack_size = round(os.path.getsize(pack_file) / 2**20, 2)

    with open(manifest_location) as mlf:
        env = yaml.safe_load(mlf.read())
    manifest = env["manifest"]
    manifest["slug"] = conda_slug
    manifest["create_date"] = datetime.utcnow().strftime(
        "%a, %b %d, %Y, %H:%M:%S %Z UTC"
    )
    manifest["size_mb"] = pack_size
    ns, bucket, prefix, _ = parse_conda_uri(conda_uri_prefix)

    pack_uri = os.path.join(
        conda_uri_prefix,
        manifest.get("arch_type", "CPU").lower(),
        manifest["name"],
        str(manifest["version"]),
        publish_slug,
    )
    manifest["pack_path"] = os.path.join(
        prefix,
        manifest.get("arch_type", "CPU").lower(),
        manifest["name"],
        str(manifest["version"]),
        publish_slug,
    )
    manifest["pack_uri"] = pack_uri
    with open(manifest_location, "w") as f:
        yaml.safe_dump(env, f)
    if pack_size > 100:
        MultiPartUploader(
            pack_file, pack_uri, 10, oci_config, oci_profile, auth_type
        ).upload(manifest)
    else:
        with open(pack_file, "rb") as pkf:
            client.put_object(
                ns,
                bucket,
                manifest["pack_path"],
                pkf,
                opc_meta={"manifest": json.dumps(manifest)},
            )
    print(f"Conda pack {pack_uri} published.")
    os.remove(pack_file)
