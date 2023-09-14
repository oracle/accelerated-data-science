#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import importlib
import inspect
import os
import re
import shutil
import tempfile
from typing import Any, Dict, Union

import fsspec
import yaml
from tabulate import tabulate

from ads.common import utils as ads_common_utils
from ads.common.decorator.runtime_dependency import (
    OptionalDependency,
    runtime_dependency,
)
from ads.opctl import logger
from ads.opctl.cmds import _BackendFactory
from ads.opctl.conda.cmds import create as conda_create
from ads.opctl.conda.cmds import publish as conda_publish
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.merger import ConfigMerger
from ads.opctl.constants import (
    BACKEND_NAME,
    DEFAULT_ADS_CONFIG_FOLDER,
    OPERATOR_MODULE_PATH,
    RESOURCE_TYPE,
    RUNTIME_TYPE,
)
from ads.opctl.operator.common.const import PACK_TYPE
from ads.opctl.operator.common.utils import OperatorInfo, _operator_info
from ads.opctl.utils import publish_image as publish_image_cmd

from .__init__ import __operators__
from .common.errors import (
    OperatorCondaNotFoundError,
    OperatorImageNotFoundError,
    OperatorNotFoundError,
)
from .common.utils import (
    _build_image,
    _load_yaml_from_uri,
    _operator_info_list,
)

OPERATOR_BASE_IMAGE = "ads-operator-base"
OPERATOR_BASE_GPU_IMAGE = "ads-operator-gpu-base"
OPERATOR_BASE_DOCKER_FILE = "Dockerfile"
OPERATOR_BASE_DOCKER_GPU_FILE = "Dockerfile.gpu"


def list() -> None:
    """Prints the list of the registered operators."""
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
    """
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()

    operator_info = {item.name: item for item in _operator_info_list()}.get(name)

    if not operator_info:
        raise OperatorNotFoundError(name)

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
    kwargs: (Dict, optional).
        Additional key value arguments.

    Returns
    -------
    Dict[Tuple, Dict]
        The dictionary where the key will be a tuple containing runtime kind and type.
        Example:
        >>> {("local","python"): {}, ("job", "container"): {}}
    """
    result = {}

    freeform_tags = {
        "operator": f"{operator_info.name}:{operator_info.version}",
    }

    # generate supported backend specifications templates YAML
    RUNTIME_TYPE_MAP = {
        RESOURCE_TYPE.JOB: [
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
        RESOURCE_TYPE.DATAFLOW: [
            {
                RUNTIME_TYPE.DATAFLOW: {
                    "conda_slug": operator_info.conda_prefix,
                    "freeform_tags": freeform_tags,
                }
            }
        ],
        BACKEND_NAME.OPERATOR_LOCAL: [
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

    for resource_type in RUNTIME_TYPE_MAP:
        for runtime_type_item in RUNTIME_TYPE_MAP[resource_type]:
            runtime_type, runtime_kwargs = next(iter(runtime_type_item.items()))

            # get config info from ini files
            p = ConfigProcessor(
                {**runtime_kwargs, **{"execution": {"backend": resource_type.value}}}
            ).step(
                ConfigMerger,
                ads_config=ads_config or DEFAULT_ADS_CONFIG_FOLDER,
                **kwargs,
            )

            uri = None
            if output:
                uri = os.path.join(
                    output,
                    f"backend_{resource_type.value.lower().replace('.','_') }"
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
                result[
                    (resource_type.value.lower(), runtime_type.value.lower())
                ] = yaml.load(yaml_str, Loader=yaml.FullLoader)

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
        If `operator` not specified.
    OperatorNotFoundError
        If `operator` not found.
    """

    # validation
    if not name:
        raise ValueError(
            f"The `name` attribute must be specified. Supported values: {__operators__}"
        )

    if name not in __operators__:
        raise OperatorNotFoundError(name)

    # generating operator specification
    operator_cmd_module = importlib.import_module(f"{OPERATOR_MODULE_PATH}.{name}.cmd")
    importlib.reload(operator_cmd_module)
    operator_specification_template = getattr(operator_cmd_module, "init")(**kwargs)

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

    # get operator physical location
    operator_path = os.path.join(os.path.dirname(__file__), "lowcode", name)

    # load operator info
    operator_info: OperatorInfo = _operator_info(operator_path)

    # save operator spec YAML
    with fsspec.open(os.path.join(output, f"{name}.yaml"), mode="w") as f:
        f.write(operator_specification_template)

    # copy README and original schema files into a destination folder
    for src_file in ("README.md", "schema.yaml", "environment.yaml"):
        ads_common_utils.copy_file(
            uri_src=os.path.join(operator_path, src_file),
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
def build_image(
    name: str = None,
    source_folder: str = None,
    gpu: bool = None,
    rebuild_base_image: bool = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Builds the image for the particular operator.
    For the service operators, the name needs to be provided.
    For the custom operators, the path (source_folder) to the operator needs to be provided.

    Parameters
    ----------
    name: (str, optional)
        Name of the operator to build the image.
    gpu: (bool, optional)
        Whether to build a GPU-enabled Docker image.
    source_folder: (str, optional)
        The folder containing the operator source code.
        Only relevant for custom operators.
    rebuild_base_image: (optional, bool)
        If rebuilding both base and operator's images required.
    kwargs: (Dict, optional).
        Additional key value arguments.

    Raises
    ------
    ValueError
        If neither `name` nor `source_folder` were provided.
    OperatorNotFoundError
        If the service operator not found.
    FileNotFoundError
        If source_folder not exists.
    """
    import docker

    operator_image_name = ""
    operator_name = name

    if name:
        if name not in __operators__:
            raise OperatorNotFoundError(name)
        source_folder = os.path.dirname(
            inspect.getfile(importlib.import_module(f"{OPERATOR_MODULE_PATH}.{name}"))
        )
        operator_image_name = operator_image_name or name
        logger.info(f"Building Docker image for the `{name}` service operator.")
    elif source_folder:
        source_folder = os.path.abspath(os.path.expanduser(source_folder))
        if not os.path.isdir(source_folder):
            raise FileNotFoundError(f"The path {source_folder} does not exist")

        operator_name = os.path.basename(source_folder.rstrip("/"))
        operator_image_name = operator_image_name or operator_name
        logger.info(
            "Building Docker image for custom operator using source folder: "
            f"`{source_folder}`."
        )
    else:
        raise ValueError(
            "No operator name or source folder specified."
            "Please provide relevant options."
        )

    # get operator details stored in operator's init file.
    operator_info: OperatorInfo = _operator_info(source_folder)
    tag = operator_info.version

    # checks if GPU base image needs to be used.
    gpu = operator_info.gpu or gpu

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    base_image_name = OPERATOR_BASE_GPU_IMAGE if gpu else OPERATOR_BASE_IMAGE

    try:
        client = docker.from_env()
        client.api.inspect_image(base_image_name)
        if rebuild_base_image:
            raise docker.errors.ImageNotFound("Rebuilding base image requested.")
    except docker.errors.ImageNotFound:
        logger.info(f"Building base operator image {base_image_name}")

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
        shutil.copytree(source_folder, os.path.join(td, "operator"))

        run_command = [
            f"FROM {base_image_name}",
            f"COPY ./operator/ $OPERATOR_DIR/{operator_name}/",
            "RUN yum install -y libX11",
        ]
        if os.path.exists(os.path.join(td, "operator", "environment.yaml")):
            run_command.append(
                f"RUN mamba env update -f $OPERATOR_DIR/{operator_name}/environment.yaml "
                "--name $CONDA_ENV_NAME && conda clean -afy"
            )

        custom_docker_file = os.path.join(td, "Dockerfile")

        with open(custom_docker_file, "w") as f:
            f.writelines("\n".join(run_command))

        result_image_name = _build_image(
            dockerfile=custom_docker_file, image_name=operator_image_name, tag=tag
        )

        logger.info(
            f"The operator image `{result_image_name}` has been successfully built."
        )


@runtime_dependency(module="docker", install_from=OptionalDependency.OPCTL)
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
        Operator's name for publishing the image.
    registry: str
        Container registry.
    ads_config: (str, optional)
        The folder where the ads opctl config located.
    kwargs: (Dict, optional).
        Additional key value arguments.

    Raises
    ------
    ValueError
        When operator's name is not provided.
    OperatorNotFoundError
        If the service operator not found.
    OperatorImageNotFoundError
        If the operator's image doesn't exist.
    """

    import docker

    if not name:
        raise ValueError(
            f"The `name` attribute must be specified. Supported values: {__operators__}"
        )

    if name not in __operators__:
        raise OperatorNotFoundError(name)

    # get operator details stored in operator's init file.
    operator_info: OperatorInfo = _operator_info(
        os.path.dirname(
            inspect.getfile(importlib.import_module(f"{OPERATOR_MODULE_PATH}.{name}"))
        )
    )

    try:
        image = f"{operator_info.name}:{operator_info.version or 'undefined'}"
        # check if the operator's image exists
        client = docker.from_env()
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
    operator_type = config.get("type", "unknown")

    if operator_type not in __operators__:
        raise OperatorNotFoundError(operator_type)

    operator_module = importlib.import_module(
        f"{OPERATOR_MODULE_PATH}.{operator_type}.operator"
    )
    operator_module.verify(config, **kwargs)


def build_conda(
    name: str = None,
    source_folder: str = None,
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
    source_folder: (str, optional)
        The folder containing the operator source code.
        Only relevant for custom operators.
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
    """
    operator_conda_name = name
    operator_name = name

    if name:
        if name not in __operators__:
            raise OperatorNotFoundError(name)
        source_folder = os.path.dirname(
            inspect.getfile(importlib.import_module(f"{OPERATOR_MODULE_PATH}.{name}"))
        )
        operator_conda_name = operator_conda_name or name
        logger.info(f"Building conda environment for the `{name}` operator.")
    elif source_folder:
        source_folder = os.path.abspath(os.path.expanduser(source_folder))
        if not os.path.isdir(source_folder):
            raise FileNotFoundError(f"The path {source_folder} does not exist")

        operator_name = os.path.basename(source_folder.rstrip("/"))
        operator_conda_name = operator_conda_name or operator_name
        logger.info(
            "Building conda environment for custom operator using source folder: "
            f"`{source_folder}`."
        )
    else:
        raise ValueError(
            "No operator name or source folder specified."
            "Please provide relevant options."
        )

    # get operator details stored in operator's __init__.py file.
    operator_info: OperatorInfo = _operator_info(source_folder)

    # invoke the conda create command
    conda_create(
        name=name,
        version=re.sub("[^0-9.]", "", operator_info.version),
        environment_file=os.path.join(source_folder, "environment.yaml"),
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
        When operator's name is not provided.
    OperatorNotFoundError
        If the service operator not found.
    OperatorCondaNotFoundError
        If the operator's image doesn't exist.
    """
    if not name:
        raise ValueError(
            f"The `name` attribute must be specified. Supported values: {__operators__}"
        )

    if name not in __operators__:
        raise OperatorNotFoundError(name)

    # get operator details stored in operator's init file.
    operator_info: OperatorInfo = _operator_info(
        os.path.dirname(
            inspect.getfile(importlib.import_module(f"{OPERATOR_MODULE_PATH}.{name}"))
        )
    )
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
