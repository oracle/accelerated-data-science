#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import re
import runpy
import shutil
import tempfile
from typing import Any, Dict, Union, Tuple

import fsspec
import yaml
from tabulate import tabulate

from ads.common import utils as ads_common_utils
from ads.common.decorator.runtime_dependency import (
    OptionalDependency,
    runtime_dependency,
)
from ads.opctl import logger
from ads.opctl.backend.ads_dataflow import DataFlowOperatorBackend
from ads.opctl.backend.ads_ml_job import MLJobOperatorBackend
from ads.opctl.backend.local import LocalOperatorBackend
from ads.opctl.cmds import _BackendFactory
from ads.opctl.conda.cmds import create as conda_create
from ads.opctl.conda.cmds import publish as conda_publish
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.merger import ConfigMerger
from ads.opctl.constants import (
    BACKEND_NAME,
    DEFAULT_ADS_CONFIG_FOLDER,
    RESOURCE_TYPE,
    RUNTIME_TYPE,
)
from ads.opctl.decorator.common import validate_environment
from ads.opctl.operator.common.const import (
    OPERATOR_BASE_DOCKER_FILE,
    OPERATOR_BASE_DOCKER_GPU_FILE,
    OPERATOR_BASE_GPU_IMAGE,
    OPERATOR_BASE_IMAGE,
    PACK_TYPE,
)
from ads.opctl.operator.common.operator_loader import OperatorInfo, OperatorLoader
from ads.opctl.utils import publish_image as publish_image_cmd

from .__init__ import __operators__
from .common.errors import OperatorCondaNotFoundError, OperatorImageNotFoundError
from .common.operator_loader import _operator_info_list
from .common.utils import _build_image


def list() -> None:
    """Prints the list of the registered service operators.

    Returns
    -------
    None
    """
    print(
        tabulate(
            (
                {
                    "Name": item.name,
                    "Version": item.version,
                    "Description": item.short_description,
                }
                for item in _operator_info_list()
            ),
            headers="keys",
        )
    )


@runtime_dependency(module="rich", install_from=OptionalDependency.OPCTL)
def info(
    name: str,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Prints the detailed information about the particular operator.

    Parameters
    ----------
    operator: str
        The name of the operator to generate the specification YAML.
    kwargs: (Dict, optional).
        Additional key value arguments.

    Returns
    -------
    None
    """
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()
    operator_info = OperatorLoader.from_uri(uri=name).load()

    console.print(
        Markdown(
            operator_info.description
            or "The description for this operator has not been specified."
        )
    )


def _init_backend_config(
    operator_info: OperatorInfo,
    ads_config: Union[str, None] = None,
    output: Union[str, None] = None,
    overwrite: bool = False,
    backend_kind: Tuple[str] = None,
    **kwargs: Dict,
):
    """
    Generates the operator's backend configs.

    Parameters
    ----------
    output: (str, optional). Defaults to None.
        The path to the folder to save the resulting specification templates.
        The Tmp folder will be created in case when `output` is not provided.
    overwrite: (bool, optional). Defaults to False.
        Whether to overwrite the result specification YAML if exists.
    ads_config: (str, optional)
        The folder where the ads opctl config located.
    backend_kind: (str, optional)
        The required backend.
    kwargs: (Dict, optional).
        Additional key value arguments.

    Returns
    -------
    Dict[Tuple, Dict]
        The dictionary where the key will be a tuple containing runtime kind and type.
        Example:
        >>> {("local","python"): {}, ("job", "container"): {}}

    Raises
    ------
    RuntimeError
        In case if the provided backend is not supported.
    """
    result = {}

    freeform_tags = {
        "operator": f"{operator_info.name}:{operator_info.version}",
    }

    # generate supported backend specifications templates YAML
    RUNTIME_TYPE_MAP = {
        RESOURCE_TYPE.JOB.value: [
            {
                RUNTIME_TYPE.PYTHON: {
                    "conda_slug": operator_info.conda
                    if operator_info.conda_type == PACK_TYPE.SERVICE
                    else operator_info.conda_prefix,
                    "freeform_tags": freeform_tags,
                }
            },
            {
                RUNTIME_TYPE.CONTAINER: {
                    "image_name": f"{operator_info.name}:{operator_info.version}",
                    "freeform_tags": freeform_tags,
                }
            },
        ],
        RESOURCE_TYPE.DATAFLOW.value: [
            {
                RUNTIME_TYPE.DATAFLOW: {
                    "conda_slug": operator_info.conda_prefix,
                    "freeform_tags": freeform_tags,
                }
            }
        ],
        BACKEND_NAME.OPERATOR_LOCAL.value: [
            {
                RUNTIME_TYPE.CONTAINER: {
                    "kind": "operator",
                    "type": operator_info.name,
                    "version": operator_info.version,
                }
            },
            {
                RUNTIME_TYPE.PYTHON: {
                    "kind": "operator",
                    "type": operator_info.name,
                    "version": operator_info.version,
                }
            },
        ],
    }

    supported_backends = tuple(
        set(RUNTIME_TYPE_MAP.keys()) & set(operator_info.backends)
    )

    if backend_kind:
        if backend_kind not in supported_backends:
            raise RuntimeError(
                f"Not supported backend - {backend_kind}. Supported backends: {supported_backends}"
            )
        supported_backends = (backend_kind,)

    for resource_type in supported_backends:
        for runtime_type_item in RUNTIME_TYPE_MAP.get(resource_type.lower(), []):
            runtime_type, runtime_kwargs = next(iter(runtime_type_item.items()))

            # get config info from ini files
            p = ConfigProcessor(
                {**runtime_kwargs, **{"execution": {"backend": resource_type}}}
            ).step(
                ConfigMerger,
                ads_config=ads_config or DEFAULT_ADS_CONFIG_FOLDER,
                **kwargs,
            )

            uri = None
            if output:
                uri = os.path.join(
                    output,
                    f"backend_{resource_type.lower().replace('.','_') }"
                    f"_{runtime_type.value.lower()}_config.yaml",
                )

            # generate YAML specification template
            yaml_str = _BackendFactory(p.config).backend.init(
                uri=uri,
                overwrite=overwrite,
                runtime_type=runtime_type.value,
                **{**kwargs, **runtime_kwargs},
            )

            if yaml_str:
                result[(resource_type.lower(), runtime_type.value.lower())] = yaml.load(
                    yaml_str, Loader=yaml.FullLoader
                )

    return result


def init(
    name: str,
    output: Union[str, None] = None,
    overwrite: bool = False,
    ads_config: Union[str, None] = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Generates a starter YAML configurations for the operator.

    Parameters
    ----------
    name: str
        The name of the operator to generate the specification YAML.
    output: (str, optional). Defaults to None.
        The path to the folder to save the resulting specification templates.
        The Tmp folder will be created in case when `output` is not provided.
    overwrite: (bool, optional). Defaults to False.
        Whether to overwrite the result specification YAML if exists.
    ads_config: (str, optional)
        The folder where the ads opctl config located.
    kwargs: (Dict, optional).
        Additional key value arguments.

    Raises
    ------
    ValueError
        If `name` not specified.
    OperatorNotFoundError
        If `operator` not found.
    """
    # validation
    if not name:
        raise ValueError(f"The `name` attribute must be specified.")

    # load operator info
    operator_info: OperatorInfo = OperatorLoader.from_uri(uri=name).load()

    # create TMP folder if one is not provided by user
    if output:
        output = os.path.join(output, "")
        if ads_common_utils.is_path_exists(uri=output) and not overwrite:
            raise ValueError(
                f"The `{output}` already exists, use `--overwrite` option if you wish to overwrite."
            )
    else:
        overwrite = True
        output = os.path.join(tempfile.TemporaryDirectory().name, "")

    # generating operator specification
    try:
        operator_cmd_module = runpy.run_module(
            f"{operator_info.name}.cmd", run_name="init"
        )
        operator_specification_template = operator_cmd_module.get("init", lambda: "")(
            **{**kwargs, **{"type": name}}
        )
        if operator_specification_template:
            with fsspec.open(
                os.path.join(output, f"{operator_info.name}.yaml"), mode="w"
            ) as f:
                f.write(operator_specification_template)
    except Exception as ex:
        logger.info(
            "The operator's specification was not generated "
            f"because it is not supported by the `{operator_info.name}` operator."
        )
        logger.debug(ex)

    # copy README and original schema files into a destination folder
    for src_file in ("README.md", "schema.yaml", "environment.yaml"):
        ads_common_utils.copy_file(
            uri_src=os.path.join(operator_info.path, src_file),
            uri_dst=output,
            force_overwrite=overwrite,
        )

    # generate supported backend specifications templates YAML
    _init_backend_config(
        operator_info=operator_info,
        ads_config=ads_config,
        output=output,
        overwrite=overwrite,
        **kwargs,
    )

    logger.info("#" * 100)
    logger.info(f"The auto-generated configs have been placed in: {output}")
    logger.info("#" * 100)


@runtime_dependency(module="docker", install_from=OptionalDependency.OPCTL)
@validate_environment
def build_image(
    name: str = None,
    rebuild_base_image: bool = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Builds the image for the particular operator.

    Parameters
    ----------
    name: (str, optional)
        Name of the operator to build the image.
    rebuild_base_image: (optional, bool)
        If rebuilding both base and operator's images required.
    kwargs: (Dict, optional).
        Additional key value arguments.

    Raises
    ------
    ValueError
        If `name` not specified.
    """
    import docker

    # validation
    if not name:
        raise ValueError(f"The `name` attribute must be specified.")

    # load operator info
    operator_info: OperatorInfo = OperatorLoader.from_uri(uri=name).load()
    logger.info(f"Building Docker image for the `{operator_info.name}` operator.")

    # checks if GPU base image needs to be used.
    gpu = operator_info.gpu

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    base_image_name = OPERATOR_BASE_GPU_IMAGE if gpu else OPERATOR_BASE_IMAGE

    try:
        client = docker.from_env()
        client.api.inspect_image(base_image_name)
        if rebuild_base_image:
            raise docker.errors.ImageNotFound("The base operator's image not found.")
    except docker.errors.ImageNotFound:
        logger.info(f"Building the base operator's image `{base_image_name}`.")

        base_docker_file = os.path.join(
            cur_dir,
            "..",
            "docker",
            "operator",
            OPERATOR_BASE_DOCKER_GPU_FILE if gpu else OPERATOR_BASE_DOCKER_FILE,
        )

        result_image_name = _build_image(
            dockerfile=base_docker_file,
            image_name=base_image_name,
            target="base",
        )

        logger.info(
            f"The base operator image `{result_image_name}` has been successfully built."
        )

    with tempfile.TemporaryDirectory() as td:
        shutil.copytree(operator_info.path, os.path.join(td, "operator"))

        run_command = [
            f"FROM {base_image_name}",
            f"COPY ./operator/ $OPERATOR_DIR/{operator_info.name}/",
            "RUN yum install -y libX11",
        ]
        if os.path.exists(os.path.join(td, "operator", "environment.yaml")):
            run_command.append(
                f"RUN mamba env update -f $OPERATOR_DIR/{operator_info.name}/environment.yaml "
                "--name $CONDA_ENV_NAME && conda clean -afy"
            )

        custom_docker_file = os.path.join(td, "Dockerfile")

        with open(custom_docker_file, "w") as f:
            f.writelines("\n".join(run_command))

        result_image_name = _build_image(
            dockerfile=custom_docker_file,
            image_name=operator_info.name,
            tag=operator_info.version,
        )

        logger.info(
            f"The operator image `{result_image_name}` has been successfully built. "
            "To publish the image to OCI Container Registry run the "
            f"`ads operator publish-image -n {result_image_name}` command"
        )


@runtime_dependency(module="docker", install_from=OptionalDependency.OPCTL)
@validate_environment
def publish_image(
    name: str,
    registry: str = None,
    ads_config: str = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Publishes operator's image to the container registry.

    Parameters
    ----------
    name: (str, optional)
        The operator or image name for publishing to container registry.
    registry: str
        Container registry.
    ads_config: (str, optional)
        The folder where the ads opctl config located.
    kwargs: (Dict, optional).
        Additional key value arguments.

    Raises
    ------
    ValueError
        If `name` not specified.
    OperatorImageNotFoundError
        If the operator's image doesn't exist.
    """

    import docker

    # validation
    if not name:
        raise ValueError(f"The `name` attribute must be specified.")

    client = docker.from_env()

    # Check if image with given name exists
    image = name
    try:
        client.api.inspect_image(image)
    except docker.errors.ImageNotFound:
        # load operator info
        operator_info: OperatorInfo = OperatorLoader.from_uri(uri=name).load()
        try:
            image = f"{operator_info.name}:{operator_info.version or 'undefined'}"
            # check if the operator's image exists
            client.api.inspect_image(image)
        except docker.errors.ImageNotFound:
            raise OperatorImageNotFoundError(operator_info.name)

    # extract registry from the ADS config.
    if not registry:
        p = ConfigProcessor().step(
            ConfigMerger,
            ads_config=ads_config or DEFAULT_ADS_CONFIG_FOLDER,
            **kwargs,
        )
        registry = p.config.get("infrastructure", {}).get("docker_registry", None)

    publish_image_cmd(
        image=image,
        registry=registry,
    )


def verify(
    config: Dict,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Verifies operator config.

    Parameters
    ----------
    config: Dict
        The operator config.
    kwargs: (Dict, optional).
        Additional key value arguments.
    """
    operator_type = config.get("type")

    # validation
    if not operator_type:
        raise ValueError(f"The `type` attribute must be specified.")

    # load operator info
    operator_info: OperatorInfo = OperatorLoader.from_uri(uri=operator_type).load()

    # validate operator
    try:
        operator_module = runpy.run_module(
            f"{operator_info.name}.operator",
            run_name="verify",
        )
        operator_module.get("verify")(config, **kwargs)
    except Exception as ex:
        print(ex)
        logger.debug(ex)
        raise ValueError(
            f"The validator is not implemented for the `{operator_info.name}` operator."
        )


def build_conda(
    name: str = None,
    conda_pack_folder: str = None,
    overwrite: bool = False,
    ads_config: Union[str, None] = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Builds the conda environment for the particular operator.
    For the service operators, the name needs to be provided.
    For the custom operators, the path (source_folder) to the operator needs to be provided.

    Parameters
    ----------
    name: str
        The name of the operator to build conda environment for..
    conda_pack_folder: str
        The destination folder to save the conda environment.
        By default will be used the path specified in the config file generated
        with `ads opctl configure` command
    overwrite: (bool, optional). Defaults to False.
        Whether to overwrite the result specification YAML if exists.
    ads_config: (str, optional)
        The folder where the ads opctl config located.
    kwargs: (Dict, optional).
        Additional key value arguments.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `name` not specified.
    """

    # validation
    if not name:
        raise ValueError(f"The `name` attribute must be specified.")

    # load operator info
    operator_info: OperatorInfo = OperatorLoader.from_uri(uri=name).load()
    logger.info(f"Building conda environment for the `{operator_info.name}` operator.")

    # invoke the conda create command
    conda_create(
        name=operator_info.name,
        version=re.sub("[^0-9.]", "", operator_info.version),
        environment_file=os.path.join(operator_info.path, "environment.yaml"),
        conda_pack_folder=conda_pack_folder,
        gpu=operator_info.gpu,
        overwrite=overwrite,
        ads_config=ads_config,
        **kwargs,
    )


def publish_conda(
    name: str = None,
    conda_pack_folder: str = None,
    overwrite: bool = False,
    ads_config: Union[str, None] = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Publishes the conda environment for the particular operator.

    Parameters
    ----------
    name: str
        The name of the operator to generate the specification YAML.
    conda_pack_folder: str
        The destination folder to save the conda environment.
        By default will be used the path specified in the config file generated
        with `ads opctl configure` command
    overwrite: (bool, optional). Defaults to False.
        Whether to overwrite the result specification YAML if exists.
    ads_config: (str, optional)
        The folder where the ads opctl config located.
    kwargs: (Dict, optional).
        Additional key value arguments.

    Raises
    ------
    ValueError
        If `name` not specified.
    OperatorCondaNotFoundError
        If the operator's conda environment not exists.
    """

    # validation
    if not name:
        raise ValueError(f"The `name` attribute must be specified.")

    # load operator info
    operator_info: OperatorInfo = OperatorLoader.from_uri(uri=name).load()

    version = re.sub("[^0-9.]", "", operator_info.version)
    slug = f"{operator_info.name}_v{version}".replace(" ", "").replace(".", "_").lower()

    # invoke the conda publish command
    try:
        conda_publish(
            slug=slug,
            conda_pack_folder=conda_pack_folder,
            overwrite=overwrite,
            ads_config=ads_config,
            **kwargs,
        )
    except FileNotFoundError:
        raise OperatorCondaNotFoundError(operator_info.name)


def create(
    name: str,
    overwrite: bool = False,
    ads_config: Union[str, None] = None,
    output: str = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Creates new operator.

    Parameters
    ----------
    name: str
        The name of the operator to generate the specification YAML.
    overwrite: (bool, optional). Defaults to False.
        Whether to overwrite the result specification YAML if exists.
    ads_config: (str, optional)
        The folder where the ads opctl config located.
    output: (str, optional). Defaults to None.
        The path to the folder to save the resulting specification templates.
        The Tmp folder will be created in case when `output` is not provided.
    kwargs: (Dict, optional).
        Additional key value arguments.
    """
    raise NotImplementedError()


def apply(config: Dict, backend: Union[Dict, str] = None, **kwargs) -> None:
    """
    Runs the operator with the given specification on the targeted backend.

    Parameters
    ----------
    config: Dict
        The operator's config.
    backend: (Union[Dict, str], optional)
        The backend config or backend name to run the operator.
    kwargs: (Dict, optional)
        Optional key value arguments to run the operator.
    """
    p = ConfigProcessor(config).step(ConfigMerger, **kwargs)

    if p.config.get("kind", "").lower() != "operator":
        raise RuntimeError("Not supported kind of workload.")

    from ads.opctl.operator import cmd as operator_cmd
    from ads.opctl.operator.common.operator_loader import (
        OperatorInfo,
        OperatorLoader,
    )

    operator_type = p.config.get("type", "").lower()

    # validation
    if not operator_type:
        raise ValueError(
            f"The `type` attribute must be specified in the operator's config."
        )

    # extracting details about the operator
    operator_info: OperatorInfo = OperatorLoader.from_uri(uri=operator_type).load()

    supported_backends = tuple(
        set(
            (
                BACKEND_NAME.JOB.value,
                BACKEND_NAME.DATAFLOW.value,
                BACKEND_NAME.OPERATOR_LOCAL.value,
                BACKEND_NAME.LOCAL.value,
            )
        )
        & set(operator_info.backends)
    )

    backend_runtime_map = {
        BACKEND_NAME.JOB.value.lower(): (
            BACKEND_NAME.JOB.value.lower(),
            RUNTIME_TYPE.PYTHON.value.lower(),
        ),
        BACKEND_NAME.DATAFLOW.value.lower(): (
            BACKEND_NAME.DATAFLOW.value.lower(),
            RUNTIME_TYPE.DATAFLOW.value.lower(),
        ),
        BACKEND_NAME.OPERATOR_LOCAL.value.lower(): (
            BACKEND_NAME.OPERATOR_LOCAL.value.lower(),
            RUNTIME_TYPE.PYTHON.value.lower(),
        ),
    }

    if not backend:
        logger.info(
            f"Backend config is not provided, the {BACKEND_NAME.LOCAL.value} "
            "will be used by default. "
        )
        backend = {"kind": BACKEND_NAME.OPERATOR_LOCAL.value}

    if isinstance(backend, str):
        backend = {
            "kind": BACKEND_NAME.OPERATOR_LOCAL.value
            if backend.lower() == BACKEND_NAME.LOCAL.value
            else backend
        }

    backend_kind = backend.get("kind").lower() or "unknown"

    # If backend kind is Job, then it is necessary to check the infrastructure kind.
    # This is necessary, because Jobs and DataFlow have similar kind,
    # The only difference would be in the infrastructure kind.
    # This is a temporary solution, the logic needs to be placed in the ConfigMerger instead.
    if backend_kind == BACKEND_NAME.JOB.value:
        if (
            backend.get("spec", {}).get("infrastructure", {}).get("type", "").lower()
            == BACKEND_NAME.DATAFLOW.value
        ):
            backend_kind = BACKEND_NAME.DATAFLOW.value

    if backend_kind not in supported_backends:
        raise RuntimeError(
            f"Not supported backend - {backend_kind}. Supported backends: {supported_backends}"
        )

    # generate backend specification in case if it is not provided
    if not backend.get("spec"):
        backends = operator_cmd._init_backend_config(
            operator_info=operator_info, backend_kind=backend_kind, **kwargs
        )
        backend = backends[backend_runtime_map[backend_kind]]

    p_backend = ConfigProcessor(
        {**backend, **{"execution": {"backend": backend_kind}}}
    ).step(ConfigMerger, **kwargs)

    p.config["runtime"] = backend
    p.config["infrastructure"] = p_backend.config["infrastructure"]
    p.config["execution"] = p_backend.config["execution"]

    if p_backend.config["execution"]["backend"].lower() in [
        BACKEND_NAME.OPERATOR_LOCAL.value,
        BACKEND_NAME.LOCAL.value,
    ]:
        if kwargs.get("dry_run"):
            logger.info(
                "The dry run option is not supported for "
                "the local backend and will be ignored."
            )
        LocalOperatorBackend(config=p.config, operator_info=operator_info).run()
    elif p_backend.config["execution"]["backend"] == BACKEND_NAME.JOB.value:
        MLJobOperatorBackend(config=p.config, operator_info=operator_info).run()
    elif p_backend.config["execution"]["backend"] == BACKEND_NAME.DATAFLOW.value:
        DataFlowOperatorBackend(config=p.config, operator_info=operator_info).run()
