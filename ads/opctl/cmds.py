#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import configparser
import click

import ads

from typing import Dict, List

from ads.common.auth import AuthContext
from ads.common.oci_datascience import DSCNotebookSession
from ads.common.extended_enum import ExtendedEnumMeta
from ads.opctl.backend.ads_ml_job import MLJobBackend, MLJobDistributedBackend
from ads.opctl.backend.ads_model_deployment import ModelDeploymentBackend
from ads.opctl.backend.local import (
    LocalBackend,
    LocalBackendDistributed,
    LocalPipelineBackend,
)
from ads.opctl.backend.ads_dataflow import DataFlowBackend
from ads.opctl.backend.ads_ml_pipeline import PipelineBackend
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.merger import ConfigMerger
from ads.opctl.config.resolver import ConfigResolver
from ads.opctl.config.utils import read_from_ini
from ads.opctl.config.validator import ConfigValidator
from ads.opctl.config.yaml_parsers import YamlSpecParser
import fsspec
from ads.common.auth import AuthType

from ads.opctl.distributed.cmds import (
    update_ini,
    increment_tag_in_ini,
    docker_build_cmd,
    update_image,
    verify_and_publish_image,
    update_config_image,
)
from ads.opctl.constants import (
    DEFAULT_OCI_CONFIG_FILE,
    DEFAULT_PROFILE,
    DEFAULT_CONDA_PACK_FOLDER,
    DEFAULT_ADS_CONFIG_FOLDER,
    ADS_CONFIG_FILE_NAME,
    ADS_JOBS_CONFIG_FILE_NAME,
    ADS_DATAFLOW_CONFIG_FILE_NAME,
    ADS_ML_PIPELINE_CONFIG_FILE_NAME,
    ADS_LOCAL_BACKEND_CONFIG_FILE_NAME,
    BACKEND_NAME,
)
from ads.opctl.utils import (
    is_in_notebook_session,
    get_service_pack_prefix,
)
import yaml


class DataScienceResource(str, metaclass=ExtendedEnumMeta):
    JOB = "datasciencejob"
    DATAFLOW = "dataflowapplication"
    PIPELINE = "datasciencepipeline"
    MODEL_DEPLOYMENT = "datasciencemodeldeployment"


class DataScienceResourceRun(str, metaclass=ExtendedEnumMeta):
    JOB_RUN = "datasciencejobrun"
    DATAFLOW_RUN = "dataflowrun"
    PIPELINE_RUN = "datasciencepipelinerun"
    MODEL_DEPLOYMENT = "datasciencemodeldeployment"


DATA_SCIENCE_RESOURCE_BACKEND_MAP = {
    DataScienceResource.JOB: "job",
    DataScienceResourceRun.JOB_RUN: "job",
    DataScienceResource.DATAFLOW: "dataflow",
    DataScienceResourceRun.DATAFLOW_RUN: "dataflow",
    DataScienceResource.PIPELINE: "pipeline",
    DataScienceResourceRun.PIPELINE_RUN: "pipeline",
    DataScienceResourceRun.MODEL_DEPLOYMENT: "deployment",
}

DATA_SCIENCE_RESOURCE_RUN_BACKEND_MAP = {
    DataScienceResourceRun.JOB_RUN: "job",
    DataScienceResourceRun.DATAFLOW_RUN: "dataflow",
    DataScienceResourceRun.PIPELINE_RUN: "pipeline",
    DataScienceResourceRun.MODEL_DEPLOYMENT: "deployment",
}


class _BackendFactory:
    BACKENDS_MAP = {
        BACKEND_NAME.JOB.value: MLJobBackend,
        BACKEND_NAME.DATAFLOW.value: DataFlowBackend,
        BACKEND_NAME.PIPELINE.value: PipelineBackend,
        BACKEND_NAME.MODEL_DEPLOYMENT.value: ModelDeploymentBackend,
    }

    LOCAL_BACKENDS_MAP = {
        BACKEND_NAME.JOB.value: LocalBackend,
        BACKEND_NAME.PIPELINE.value: LocalPipelineBackend,
    }

    def __init__(self, config: Dict):
        self.config = config
        self._backend = config["execution"].pop("backend", None)
        if self._backend is None:
            raise RuntimeError("Please specify backend.")
        elif (
            self._backend != BACKEND_NAME.LOCAL.value
            and self._backend not in self.BACKENDS_MAP
        ):
            raise NotImplementedError(f"backend {self._backend} is not implemented.")

    @property
    def backend(self):
        if self._backend == BACKEND_NAME.LOCAL.value:
            kind = self.config.get("kind")
            if kind not in self.LOCAL_BACKENDS_MAP:
                options = [backend for backend in self.LOCAL_BACKENDS_MAP.keys()]
                # Special case local backend option not supported by this factory.
                options.append("distributed")
                raise RuntimeError(
                    f"kind {kind} not supported by local backend. Please choose from: "
                    f"[{str.join('|', options)}]"
                )
            return self.LOCAL_BACKENDS_MAP[kind](self.config)

        return self.BACKENDS_MAP[self._backend](self.config)


def _save_yaml(yaml_content, **kwargs):
    """Saves job run info YAML to a local file.

    Parameters
    ----------
    yaml_content : str
        YAML content as string.
    """
    if kwargs["job_info"]:
        yaml_path = os.path.abspath(os.path.expanduser(kwargs["job_info"]))
        if os.path.isfile(yaml_path):
            overwrite = input(
                f"File {yaml_path} already exists. Overwrite the file? [yN]: "
            )
            if overwrite not in ["y", "Y"]:
                return
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"Job run info saved to {yaml_path}")


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
    if config.get("kind") == "distributed":  # TODO: add kind factory
        print(
            "......................... Initializing the process ..................................."
        )
        ini = update_ini(
            kwargs["tag"],
            kwargs["registry"],
            kwargs["dockerfile"],
            kwargs["source_folder"],
            config,
            kwargs["nobuild"],
        )
        nobuild = kwargs["nobuild"]
        mode = kwargs["backend"]
        increment = kwargs["auto_increment"]

        if not nobuild:
            if increment:
                ini = increment_tag_in_ini(ini)
            docker_build_cmd(ini)
        config = update_image(config, ini)

        if mode == BACKEND_NAME.LOCAL.value:
            print(
                "\u26A0 Docker Image: "
                + ini.get("main", "registry")
                + ":"
                + ini.get("main", "tag")
                + " is not pushed to oci artifacts."
            )
            print("running image: " + config["spec"]["cluster"]["spec"]["image"])

            backend = LocalBackendDistributed(config)
            backend.run()
        elif mode == BACKEND_NAME.DATAFLOW.value:
            raise RuntimeError(
                "backend operator for distributed training can either be local or job"
            )
        else:
            if not kwargs["dry_run"]:
                verify_and_publish_image(kwargs["nopush"], config)
                print("running image: " + config["spec"]["cluster"]["spec"]["image"])
            cluster_def = YamlSpecParser.parse_content(config)

            backend = MLJobDistributedBackend(p.config)

            # Define job first,
            # Then Run
            cluster_run_info = backend.run(
                cluster_info=cluster_def, dry_run=p.config["execution"].get("dry_run")
            )
            if cluster_run_info:
                cluster_run = {}
                cluster_run["jobId"] = cluster_run_info[0].id
                cluster_run["workDir"] = cluster_def.cluster.work_dir
                cluster_run["mainJobRunId"] = {
                    cluster_run_info[1].name: cluster_run_info[1].id
                }
                if len(cluster_run_info[2]) > 0:
                    cluster_run["otherJobRunIds"] = [
                        {wj.name: wj.id} for wj in cluster_run_info[2]
                    ]
                yamlContent = yaml.dump(cluster_run)
                yamlContent += (
                    "# \u2b50 To stream the logs of the main job run:\n"
                    + f"# \u0024 ads opctl watch {list(cluster_run['mainJobRunId'].values())[0]}"
                )
                print(yamlContent)
                _save_yaml(yamlContent, **kwargs)
            return cluster_run_info
    else:
        if (
            "kind" in p.config
            and p.config["execution"].get("backend", None) != BACKEND_NAME.LOCAL.value
        ):
            p.config["execution"]["backend"] = p.config["kind"]
            return _BackendFactory(p.config).backend.apply()

        if "ocid" in p.config["execution"]:
            resource_to_backend = {
                DataScienceResource.JOB: BACKEND_NAME.JOB,
                DataScienceResource.DATAFLOW: BACKEND_NAME.DATAFLOW,
                DataScienceResource.PIPELINE: BACKEND_NAME.PIPELINE,
            }
            for r, b in resource_to_backend.items():
                if r in p.config["execution"]["ocid"]:
                    p.config["execution"]["backend"] = b.value
        else:
            p.step(ConfigResolver).step(ConfigValidator)
            # spec may have changed during validation step (e.g. defaults filled in)
            # thus command need to be updated since it encodes spec
            p = ConfigResolver(p.config)
            p._resolve_command()
        return _BackendFactory(p.config).backend.run()


def run_diagnostics(config: Dict, **kwargs) -> Dict:
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
    if config.get("kind") == "distributed":  # TODO: add kind factory

        config = update_config_image(config)
        cluster_def = YamlSpecParser.parse_content(config)

        backend = MLJobDistributedBackend(p.config)

        # Define job first,
        # Then Run
        cluster_run_info = backend.run_diagnostics(
            cluster_info=cluster_def, dry_run=p.config["execution"].get("dry_run")
        )
        if cluster_run_info:
            diagnostics_report_path = os.path.join(
                cluster_def.cluster.work_dir,
                cluster_run_info[0].id,
                "diagnostic_report.html",
            )
            with fsspec.open(
                diagnostics_report_path, "r", **backend.oci_auth
            ) as infile:
                with open(kwargs["output"], "w") as outfile:
                    outfile.write(infile.read())
        return cluster_run_info
    else:
        print("Diagnostics not available for kind: {config.get('kind')}")


def _update_env_vars(config, env_vars: List):
    """
    env_vars: List, should be formatted as [{"name": "OCI__XXX", "value": YYY},]
    """
    # TODO move this to a class which checks the version, kind, type, etc.
    config["spec"]["Runtime"]["spec"]["environmentVariables"].extend(env_vars)
    return config


def init_operator(**kwargs) -> str:
    """
    Initialize the resources for an operator

    Parameters
    ----------
    kwargs: dict
        keyword argument, stores command line args
    Returns
    -------
    folder_path: str
        a path to the folder with all of the resources
    """
    # TODO: confirm that operator slug is in the set of valid operator slugs
    assert kwargs["operator_slug"] == "dask_cluster"

    if kwargs.get("folder_path"):
        kwargs["operator_folder_path"] = kwargs.pop("folder_path")[0]
    else:
        kwargs["operator_folder_path"] = kwargs["operator_slug"]
    p = ConfigProcessor().step(ConfigMerger, **kwargs)
    print(f"config check: {p.config}")
    return _BackendFactory(p.config).backend.init_operator()


def delete(**kwargs) -> None:
    """
    Delete a MLJob/DataFlow run.

    Parameters
    ----------
    kwargs: dict
        keyword argument, stores command line args
    Returns
    -------
    None
    """
    kwargs["backend"] = _get_backend_from_ocid(kwargs["ocid"])

    if (
        DataScienceResourceRun.JOB_RUN in kwargs["ocid"]
        or DataScienceResourceRun.DATAFLOW_RUN in kwargs["ocid"]
        or DataScienceResourceRun.PIPELINE_RUN in kwargs["ocid"]
        or DataScienceResourceRun.MODEL_DEPLOYMENT in kwargs["ocid"]
    ):
        kwargs["run_id"] = kwargs.pop("ocid")
    elif (
        DataScienceResource.JOB in kwargs["ocid"]
        or DataScienceResource.DATAFLOW in kwargs["ocid"]
        or DataScienceResource.PIPELINE in kwargs["ocid"]
    ):
        kwargs["id"] = kwargs.pop("ocid")
    else:
        raise ValueError(f"{kwargs['ocid']} is valid or supported.")

    p = ConfigProcessor().step(ConfigMerger, **kwargs)
    return _BackendFactory(p.config).backend.delete()


def cancel(**kwargs) -> None:
    """
    Cancel a MLJob/DataFlow run.

    Parameters
    ----------
    kwargs: dict
        keyword argument, stores command line args
    Returns
    -------
    None
    """
    kwargs["run_id"] = kwargs.pop("ocid")
    if not kwargs.get("backend"):
        kwargs["backend"] = _get_backend_from_run_id(kwargs["run_id"])
    p = ConfigProcessor().step(ConfigMerger, **kwargs)
    return _BackendFactory(p.config).backend.cancel()


def watch(**kwargs) -> None:
    """
    Watch a MLJob/DataFlow run.

    Parameters
    ----------
    kwargs: dict
        keyword argument, stores command line args
    Returns
    -------
    None
    """
    kwargs["run_id"] = kwargs.pop("ocid")
    if not kwargs.get("backend"):
        kwargs["backend"] = _get_backend_from_run_id(kwargs["run_id"])
    p = ConfigProcessor().step(ConfigMerger, **kwargs)
    return _BackendFactory(p.config).backend.watch()


def activate(**kwargs) -> None:
    """
    Activate a ModelDeployment.

    Parameters
    ----------
    kwargs: dict
        keyword argument, stores command line args

    Returns
    -------
    None
    """
    kwargs["run_id"] = kwargs.pop("ocid")
    if not kwargs.get("backend"):
        kwargs["backend"] = _get_backend_from_run_id(kwargs["run_id"])
    p = ConfigProcessor().step(ConfigMerger, **kwargs)
    return _BackendFactory(p.config).backend.activate()


def deactivate(**kwargs) -> None:
    """
    Deactivate a ModelDeployment.

    Parameters
    ----------
    kwargs: dict
        keyword argument, stores command line args

    Returns
    -------
    None
    """
    kwargs["run_id"] = kwargs.pop("ocid")
    if not kwargs.get("backend"):
        kwargs["backend"] = _get_backend_from_run_id(kwargs["run_id"])
    p = ConfigProcessor().step(ConfigMerger, **kwargs)
    return _BackendFactory(p.config).backend.deactivate()


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
    p._resolve_mounted_volumes()
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
        config_parser = read_from_ini(os.path.join(folder, ADS_CONFIG_FILE_NAME))
    else:
        config_parser = configparser.ConfigParser(default_section=None)
        config_parser.optionxform = str

    if "OCI" not in config_parser:
        config_parser["OCI"] = {}
    if "CONDA" not in config_parser:
        config_parser["CONDA"] = {}

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
    conda_pack_path = click.prompt(
        "Conda pack install folder:",
        default=config_parser["CONDA"].get(
            "conda_pack_folder", DEFAULT_CONDA_PACK_FOLDER
        ),
    )
    config_parser["CONDA"]["conda_pack_folder"] = os.path.abspath(
        os.path.expanduser(conda_pack_path)
    )
    if is_in_notebook_session():
        conda_os_prefix_default = config_parser["CONDA"].get(
            "conda_pack_os_prefix", get_service_pack_prefix()
        )
    else:
        conda_os_prefix_default = config_parser["CONDA"].get("conda_pack_os_prefix", "")

    conda_os_prefix = click.prompt(
        "Object storage Conda Env prefix, in the format oci://<bucket>@<namespace>/<path>",
        default=conda_os_prefix_default,
    )
    config_parser["CONDA"]["conda_pack_os_prefix"] = conda_os_prefix

    with open(os.path.join(os.path.expanduser(folder), ADS_CONFIG_FILE_NAME), "w") as f:
        config_parser.write(f)
    print(f"Configuration saved at {os.path.join(folder, ADS_CONFIG_FILE_NAME)}")

    print("==== Setting configuration for OCI Jobs ====")
    if click.confirm(
        f"Do you want to set up or update OCI Jobs configuration?", default=True
    ):
        required_fields = [
            ("compartment_id", ""),
            ("project_id", ""),
            ("subnet_id", ""),
            ("shape_name", ""),
            ("block_storage_size_in_GBs", ""),
        ]

        optional_fields = [
            ("log_group_id", ""),
            ("log_id", ""),
            ("docker_registry", ""),
            ("conda_pack_os_prefix", "in the format oci://<bucket>@<namespace>/<path>"),
        ]
        _set_service_configurations(
            ADS_JOBS_CONFIG_FILE_NAME,
            required_fields,
            optional_fields,
            folder,
            oci_config_path,
            is_in_notebook_session(),
        )

    print("==== Setting configuration for OCI DataFlow ====")
    if click.confirm(
        f"Do you want to set up or update OCI DataFlow configuration?", default=True
    ):
        required_fields = [
            ("compartment_id", ""),
            ("driver_shape", ""),
            ("executor_shape", ""),
            ("logs_bucket_uri", ""),
            ("script_bucket", "in the format oci://<bucket>@<namespace>/<path>"),
        ]

        optional_fields = [
            ("num_executors", ""),
            ("spark_version", ""),
            ("archive_bucket", "in the format oci://<bucket>@<namespace>/<path>"),
        ]
        _set_service_configurations(
            ADS_DATAFLOW_CONFIG_FILE_NAME,
            required_fields,
            optional_fields,
            folder,
            oci_config_path,
            is_in_notebook_session(),
        )

    print("==== Setting configuration for OCI ML Pipeline ====")
    if click.confirm(
        f"Do you want to set up or update OCI ML Pipeline configuration?", default=True
    ):
        required_fields = [
            ("compartment_id", ""),
            ("project_id", ""),
        ]

        optional_fields = [
            ("log_group_id", ""),
            ("log_id", ""),
        ]
        _set_service_configurations(
            ADS_ML_PIPELINE_CONFIG_FILE_NAME,
            required_fields,
            optional_fields,
            folder,
            oci_config_path,
            is_in_notebook_session(),
        )

    print("==== Setting configuration for local backend ====")
    if click.confirm(
        f"Do you want to set up or update local backend configuration?", default=True
    ):
        required_fields = [
            ("max_parallel_containers", str(min(os.cpu_count(), 4))),
            ("pipeline_status_poll_interval_seconds", str(5)),
        ]

        optional_fields = []

        _set_service_configurations(
            ADS_LOCAL_BACKEND_CONFIG_FILE_NAME,
            required_fields,
            optional_fields,
            folder,
            oci_config_path,
            is_in_notebook_session(),
        )


def _set_service_configurations(
    config_file_name,
    required_fields,
    optional_fields,
    config_folder,
    oci_config_path=None,
    rp=False,
):
    if os.path.exists(os.path.join(config_folder, config_file_name)):
        infra_parser = read_from_ini(os.path.join(config_folder, config_file_name))
    else:
        infra_parser = configparser.ConfigParser(default_section=None)
        infra_parser.optionxform = str

    def prompt_for_values(infra_parser, profile):
        for field, msg in required_fields:
            prompt = f"Specify {field}" if len(msg) == 0 else f"Specify {field}, {msg}"
            infra_parser[profile][field] = click.prompt(
                prompt,
                default=infra_parser[profile].get(field, None),
            )
        for field, msg in optional_fields:
            prompt = f"Specify {field}" if len(msg) == 0 else f"Specify {field}, {msg}"
            ans = click.prompt(
                prompt,
                default=infra_parser[profile].get(field, ""),
            )
            if len(ans) > 0:
                infra_parser[profile][field] = ans

    if rp:
        if "RESOURCE_PRINCIPAL" not in infra_parser:
            with AuthContext():
                ads.set_auth(auth=AuthType.RESOURCE_PRINCIPAL)
                notebook_session = DSCNotebookSession.from_ocid(
                    os.environ["NB_SESSION_OCID"]
                )
            default_values = {
                "compartment_id": notebook_session.compartment_id,
                "project_id": notebook_session.project_id,
                "subnet_id": notebook_session.notebook_session_configuration_details.subnet_id,
                "block_storage_size_in_GBs": notebook_session.notebook_session_configuration_details.block_storage_size_in_gbs,
                "shape_name": notebook_session.notebook_session_configuration_details.shape,
                "driver_shape": notebook_session.notebook_session_configuration_details.shape,
                "executor_shape": notebook_session.notebook_session_configuration_details.shape,
            }
            infra_parser["RESOURCE_PRINCIPAL"] = {
                k: v
                for k, v in default_values.items()
                if (
                    k in [field[0] for field in required_fields]
                    or k in [field[0] for field in optional_fields]
                )
            }
        if click.confirm(
            "Do you want to set up or update configuration when using resource principal?",
            default=True,
        ):
            prompt_for_values(infra_parser, "RESOURCE_PRINCIPAL")

    if os.path.exists(os.path.expanduser(oci_config_path)):
        parser = read_from_ini(os.path.abspath(os.path.expanduser(oci_config_path)))
        for oci_profile in parser:
            if oci_profile:
                if click.confirm(
                    f"Do you want to set up or update for profile {oci_profile}?"
                ):
                    if oci_profile not in infra_parser:
                        infra_parser[oci_profile] = {}
                    prompt_for_values(infra_parser, oci_profile)

    with open(os.path.join(config_folder, config_file_name), "w") as f:
        infra_parser.write(f)
    print(f"Configuration saved at {os.path.join(config_folder, config_file_name)}")


def _get_backend_from_ocid(ocid: str) -> str:
    for value in DataScienceResource.values():
        if value in ocid:
            return DATA_SCIENCE_RESOURCE_BACKEND_MAP[value]
    raise ValueError("Must provide a resource OCID.")


def _get_backend_from_run_id(ocid: str) -> str:
    for value in DataScienceResourceRun.values():
        if value in ocid:
            return DATA_SCIENCE_RESOURCE_RUN_BACKEND_MAP[value]
    raise ValueError("Must provide a resource run OCID.")
