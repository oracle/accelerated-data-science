#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import importlib
import inspect
import os
import shutil
import tempfile
from typing import Any, Dict, Union

import fsspec
from tabulate import tabulate

from ads.common import utils as ads_common_utils
from ads.common.decorator.runtime_dependency import (
    OptionalDependency,
    runtime_dependency,
)
from ads.opctl import logger
from ads.opctl.cmds import _BackendFactory
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.merger import ConfigMerger
from ads.opctl.constants import (
    DEFAULT_ADS_CONFIG_FOLDER,
    OPERATOR_MODULE_PATH,
    RESOURCE_TYPE,
    RUNTIME_TYPE,
    BACKEND_NAME,
)
from ads.opctl.operator.common.utils import OperatorInfo, _operator_info
from ads.opctl.utils import publish_image as publish_image_cmd

from .__init__ import __operators__
from .common.utils import (
    _build_image,
    _convert_schema_to_html,
    _load_yaml_from_uri,
    _operator_info_list,
)

OPERATOR_BASE_IMAGE = "ads-operator-base"
OPERATOR_BASE_GPU_IMAGE = "ads-operator-gpu-base"
OPERATOR_BASE_DOCKER_FILE = "Dockerfile"
OPERATOR_BASE_DOCKER_GPU_FILE = "Dockerfile.gpu"


class OperatorNotFoundError(Exception):
    def __init__(self, operator: str):
        super().__init__(
            f"Operator with name: `{operator}` is not found."
            "Use `ads opctl operator list` to get the list of registered operators."
        )


def list() -> None:
    """
    Prints the list of the registered operators.
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
    """
    operator_info = {item.name: item for item in _operator_info_list()}.get(name)
    if operator_info:
        print(operator_info.description)
    else:
        raise OperatorNotFoundError(name)


def init(
    name: str,
    output: Union[str, None] = None,
    overwrite: bool = False,
    ads_config: Union[str, None] = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Generates a starter specification template YAML for the operator.

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
        Any optional kwargs arguments.

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

    # save operator schema in HTML format
    module_schema = _load_yaml_from_uri(os.path.join(operator_path, "schema.yaml"))
    with fsspec.open(os.path.join(output, "schema.html"), mode="w") as f:
        f.write(_convert_schema_to_html(name, module_schema))

    # copy README and original schema files into a destination folder
    for src_file in ("README.md", "schema.yaml"):
        ads_common_utils.copy_file(
            uri_src=os.path.join(operator_path, src_file),
            uri_dst=output,
            force_overwrite=overwrite,
        )

    # generate supported backend specifications templates YAML
    RUNTIME_TYPE_MAP = {
        RESOURCE_TYPE.JOB: [
            {RUNTIME_TYPE.PYTHON: {"conda_slug": operator_info.conda}},
            {
                RUNTIME_TYPE.CONTAINER: {
                    "image_name": f"{operator_info.name}:{operator_info.version}"
                }
            },
        ],
        RESOURCE_TYPE.DATAFLOW: [{RUNTIME_TYPE.DATAFLOW: {}}],
        BACKEND_NAME.OPERATOR_LOCAL: [
            {
                RUNTIME_TYPE.CONTAINER: {
                    "kind": "operator",
                    "type": operator_info.name,
                    "version": operator_info.version,
                }
            }
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

            # generate YAML specification template
            _BackendFactory(p.config).backend.init(
                uri=os.path.join(
                    output,
                    f"backend_{resource_type.value.lower().replace('.','_') }"
                    f"_{runtime_type.value.lower()}_config.yaml",
                ),
                overwrite=overwrite,
                runtime_type=runtime_type.value,
                **{**kwargs, **runtime_kwargs},
            )

    print("#" * 100)
    print(f"The auto-generated configs location: {output}")
    print("#" * 100)


@runtime_dependency(module="docker", install_from=OptionalDependency.OPCTL)
def build_image(
    name: str = None,
    source_folder: str = None,
    image: str = None,
    tag: str = None,
    gpu: bool = None,
    rebuild_base_image: bool = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Builds image for the operator.
    For the built-in operators, the name needs to be provided.
    For the custom operators, the path (source_folder) to the operator needs to be provided.

    Parameters
    ----------
    name: (str, optional)
        Name of the service operator to build the image.
        Only relevant for built-in service operators.
    gpu: (bool, optional)
        Whether to build a GPU-enabled Docker image.
    source_folder: (str, optional)
        The folder containing the operator source code.
        Only relevant for custom operators.
    image: (optional, str)
        The name of the image. The operator name will be used if not provided.
    tag: (optional, str)
       The tag of the image. The `latest` will be used if not provided.
    rebuild_base_image: (optional, bool)
        If rebuilding both base and operator's images required.
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

    operator_image_name = image
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


def publish_image(
    image: str,
    registry: str = None,
    ads_config: str = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Published image to the container registry.

    Parameters
    ----------
    image: str
        The name of the image.
    registry: str
        Container registry.
    ads_config: (str, optional)
        The folder where the ads opctl config located.
    kwargs: (Dict, optional).
        Any optional kwargs arguments.
    """
    if not image:
        raise ValueError("To publish image, the image name needs to be provided.")

    if not registry:
        p = ConfigProcessor().step(
            ConfigMerger,
            ads_config=ads_config or DEFAULT_ADS_CONFIG_FOLDER,
            **kwargs,
        )
        registry = p.config.get("infrastructure", {}).get("docker_registry", None)

    publish_image_cmd(image=image, registry=registry)


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
        Any optional kwargs arguments.
    """
    operator_type = config.get("type", "unknown")

    if operator_type not in __operators__:
        raise OperatorNotFoundError(operator_type)

    operator_module = importlib.import_module(
        f"{OPERATOR_MODULE_PATH}.{operator_type}.operator"
    )
    operator_module.verify(config)


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
        Any optional kwargs arguments.
    """
    raise NotImplementedError()
