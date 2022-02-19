#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import configparser
import os
from typing import Dict

import click

from ads.opctl.backend.ads_ml_job import MLJobBackend
from ads.opctl.backend.local import LocalBackend
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.merger import ConfigMerger
from ads.opctl.config.resolver import ConfigResolver
from ads.opctl.config.utils import _read_from_ini
from ads.opctl.config.validator import ConfigValidator
from ads.opctl.constants import (
    OPS_IMAGE_BASE,
    OPS_IMAGE_GPU_BASE,
    ML_JOB_IMAGE,
    ML_JOB_GPU_IMAGE,
    DEFAULT_OCI_CONFIG_FILE,
    DEFAULT_PROFILE,
    DEFAULT_CONDA_PACK_FOLDER,
    CONDA_PACK_OS_PREFIX_FORMAT,
    DEFAULT_ADS_CONFIG_FOLDER,
    ADS_CONFIG_FILE_NAME,
    ADS_JOBS_CONFIG_FILE_NAME,
)


class _BackendFactory:

    BACKENDS_MAP = {"local": LocalBackend, "job": MLJobBackend}

    def __init__(self, config: Dict):
        self.config = config
        self._backend = config["execution"].pop("backend", None)
        if self._backend is None:
            raise RuntimeError("Please specify backend.")
        elif self._backend not in self.BACKENDS_MAP:
            raise NotImplementedError(f"backend {self._backend} is not implemented.")

    @property
    def backend(self):
        return self.BACKENDS_MAP[self._backend](self.config)


def run(config: Dict, **kwargs) -> Dict:
    """
    Run a job given configuration and command line args passed in (kwargs).

    Parameters
    ----------
    config: dict
        dictionary of configurations
    kwargs: dict
        keyword arguments, stores configuration from command line args

    Returns
    -------
    Dict
        dictionary of job id and run id in case of ML Job run, else empty if running locally
    """
    p = ConfigProcessor(config).step(ConfigMerger, **kwargs)
    if p.config["execution"].get("job_id", None):
        p.config["execution"]["backend"] = "job"
        return _BackendFactory(p.config).backend.run()
    p.step(ConfigResolver).step(ConfigValidator)
    # spec may have changed during validation step (e.g. defaults filled in)
    # thus command need to be updated since it encodes spec
    p = ConfigResolver(p.config)
    p._resolve_command()
    return _BackendFactory(p.config).backend.run()


def delete(**kwargs) -> None:
    """
    Delete a ML Job or a ML Job run.

    Parameters
    ----------
    kwargs: dict
        keyword argument, stores command line args
    Returns
    -------
    None
    """
    if "datasciencejobrun" in kwargs["ocid"]:
        kwargs["run_id"] = kwargs.pop("ocid")
    elif "datasciencejob" in kwargs["ocid"]:
        kwargs["job_id"] = kwargs.pop("ocid")
    else:
        raise ValueError(f"{kwargs['ocid']} is not a job run or job id.")
    kwargs["backend"] = "job"
    p = ConfigProcessor().step(ConfigMerger, **kwargs)
    return _BackendFactory(p.config).backend.delete()


def cancel(**kwargs) -> None:
    """
    Cancel a ML Job run.

    Parameters
    ----------
    kwargs: dict
        keyword argument, stores command line args
    Returns
    -------
    None
    """
    kwargs["run_id"] = kwargs.pop("ocid") or kwargs.pop("run_id")
    if "datasciencejobrun" not in kwargs["run_id"]:
        raise ValueError("Must provide a job run OCID.")
    kwargs["backend"] = "job"
    p = ConfigProcessor().step(ConfigMerger, **kwargs)
    return _BackendFactory(p.config).backend.cancel()


def watch(**kwargs) -> None:
    """
    Watch a ML Job run.

    Parameters
    ----------
    kwargs: dict
        keyword argument, stores command line args
    Returns
    -------
    None
    """
    kwargs["run_id"] = kwargs.pop("ocid") or kwargs.pop("run_id")
    if "datasciencejobrun" not in kwargs["run_id"]:
        raise ValueError("Must provide a job run OCID.")
    kwargs["backend"] = "job"
    p = ConfigProcessor().step(ConfigMerger, **kwargs)
    return _BackendFactory(p.config).backend.watch()


def init_vscode(**kwargs) -> None:
    """
    Create a .devcontainer.json file for local development.

    Parameters
    ----------
    kwargs
        keyword arguments, stores command line args
    Returns
    -------
    None
    """
    p = ConfigMerger({}).process(**kwargs)
    p = ConfigResolver(p.config)
    p._resolve_env_vars()
    p._reslve_mounted_volumes()
    use_gpu = p.config["execution"].get("gpu", False)
    LocalBackend(p.config).init_vscode_container()


def configure() -> None:
    """
    Save default configurations for opctl.

    Parameters
    ----------
    ml_job: bool
        turn on/off saving configurations for ML Job

    Returns
    -------
    None
    """
    folder = os.path.abspath(
        os.path.expanduser(
            click.prompt(
                "Folder to save ADS operators related configurations:",
                default=DEFAULT_ADS_CONFIG_FOLDER,
            )
        )
    )
    os.makedirs(os.path.expanduser(folder), exist_ok=True)

    if os.path.exists(os.path.join(folder, ADS_CONFIG_FILE_NAME)):
        config_parser = _read_from_ini(os.path.join(folder, ADS_CONFIG_FILE_NAME))
    else:
        config_parser = configparser.ConfigParser(default_section=None)
        config_parser.optionxform = str
    if "OCI" not in config_parser:
        config_parser["OCI"] = {}
    oci_config_path = click.prompt(
        "OCI config path:",
        default=config_parser["OCI"].get("oci_config", DEFAULT_OCI_CONFIG_FILE),
    )
    oci_profile = click.prompt(
        "Default OCI profile:",
        default=config_parser["OCI"].get("oci_profile", DEFAULT_PROFILE),
    )
    oci_config_parser = configparser.ConfigParser()
    oci_config_parser.read(os.path.expanduser(oci_config_path))
    if oci_profile not in oci_config_parser:
        raise ValueError(f"profile {oci_profile} is not found in {oci_config_path}.")
    config_parser["OCI"] = {
        "oci_config": oci_config_path,
        "oci_profile": oci_profile,
    }
    if "CONDA" not in config_parser:
        config_parser["CONDA"] = {}
    conda_pack_path = click.prompt(
        "Conda pack install folder:",
        default=config_parser["CONDA"].get(
            "conda_pack_folder", DEFAULT_CONDA_PACK_FOLDER
        ),
    )

    config_parser["CONDA"] = {
        "conda_pack_folder": os.path.abspath(os.path.expanduser(conda_pack_path)),
    }

    with open(os.path.join(os.path.expanduser(folder), ADS_CONFIG_FILE_NAME), "w") as f:
        config_parser.write(f)
    print(f"Configuration saved at {os.path.join(folder, ADS_CONFIG_FILE_NAME)}")

    required_fields = [
        "compartment_id",
        "project_id",
        "subnet_id",
        "shape_name",
        "block_storage_size_in_GBs",
    ]

    optional_fields = [
        "log_group_id",
        "log_id",
        "docker_registry",
    ]

    if os.path.exists(os.path.join(folder, ADS_JOBS_CONFIG_FILE_NAME)):
        infra_parser = _read_from_ini(os.path.join(folder, ADS_JOBS_CONFIG_FILE_NAME))
    else:
        infra_parser = configparser.ConfigParser(default_section=None)
        infra_parser.optionxform = str

    if not click.confirm(
        f"Do you want to set up or update OCI Jobs configuration? \nNeed {required_fields + optional_fields + ['conda_pack_os_prefix']}"
    ):
        return
    print("==== Setting configuration for OCI Jobs ====")
    parser = _read_from_ini(os.path.abspath(os.path.expanduser(oci_config_path)))
    for oci_profile in parser:
        if oci_profile:
            if click.confirm(
                f"Do you want to set up or update for profile {oci_profile}?"
            ):
                if oci_profile not in infra_parser:
                    infra_parser[oci_profile] = {}
                for field in required_fields:
                    infra_parser[oci_profile][field] = click.prompt(
                        f"Specify {field}",
                        default=infra_parser[oci_profile].get(field, None),
                    )
                for field in optional_fields:
                    ans = click.prompt(
                        f"Specify {field}",
                        default=infra_parser[oci_profile].get(field, ""),
                    )
                    if len(ans) > 0:
                        infra_parser[oci_profile][field] = ans
                ans = click.prompt(
                    "Specify object storage path to publish conda pack e.g. oci://<bucket>@<namespace>/conda-packs/)",
                    default=infra_parser[oci_profile].get("conda_pack_os_prefix", ""),
                )
                if len(ans) > 0:
                    infra_parser[oci_profile]["conda_pack_os_prefix"] = ans

    with open(os.path.join(folder, ADS_JOBS_CONFIG_FILE_NAME), "w") as f:
        infra_parser.write(f)
    print(f"Configuration saved at {os.path.join(folder, ADS_JOBS_CONFIG_FILE_NAME)}")
