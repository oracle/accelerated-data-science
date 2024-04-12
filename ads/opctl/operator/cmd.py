#!/usr/bin/env python

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import re
import runpy
import shutil
import tempfile
from typing import Any, Dict, Union

import fsspec
import yaml
from ads.opctl.operator.common.utils import print_traceback
from tabulate import tabulate

from ads.common import utils as ads_common_utils
from ads.common.decorator.runtime_dependency import (
    OptionalDependency,
    runtime_dependency,
)
from ads.opctl import logger
from ads.opctl.conda.cmds import create as conda_create
from ads.opctl.conda.cmds import publish as conda_publish
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.merger import ConfigMerger
from ads.opctl.constants import DEFAULT_ADS_CONFIG_FOLDER
from ads.opctl.decorator.common import validate_environment
from ads.opctl.operator.common.const import (
    OPERATOR_BASE_DOCKER_FILE,
    OPERATOR_BASE_DOCKER_GPU_FILE,
    OPERATOR_BASE_GPU_IMAGE,
    OPERATOR_BASE_IMAGE,
    OPERATOR_BACKEND_SECTION_NAME,
)
from ads.opctl.operator.common.operator_loader import OperatorInfo, OperatorLoader
from ads.opctl.utils import publish_image as publish_image_cmd

from .__init__ import __operators__
from .common import utils as operator_utils
from .common.backend_factory import BackendFactory
from .common.errors import (
    OperatorCondaNotFoundError,
    OperatorImageNotFoundError,
    InvalidParameterError,
)
from .common.operator_loader import _operator_info_list


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
                    "Type": item.type,
                    "Version": item.version,
                    "Description": item.description,
                }
                for item in _operator_info_list()
            ),
            headers="keys",
        )
    )


@runtime_dependency(module="rich", install_from=OptionalDependency.OPCTL)
def info(
    type: str,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Prints the detailed information about the particular operator.

    Parameters
    ----------
    type: str
        The type of the operator to get detailed.
    kwargs: (Dict, optional).
        Additional key value arguments.

    Returns
    -------
    None
    """
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()
    operator_info = OperatorLoader.from_uri(uri=type).load()

    operator_readme = None
    if operator_info.path:
        readme_file_path = os.path.join(operator_info.path, "README.md")

        if os.path.exists(readme_file_path):
            with open(readme_file_path, "r") as readme_file:
                operator_readme = readme_file.read()

    console.print(
        Markdown(
            operator_readme
            or operator_info.description
            or "The description for this operator has not been specified."
        )
    )


def init(
    type: str,
    output: Union[str, None] = None,
    overwrite: bool = False,
    merge_config: bool = False,
    ads_config: Union[str, None] = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Generates a starter YAML configurations for the operator.

    Parameters
    ----------
    type: str
        The type of the operator to generate the specification YAML.
    output: (str, optional). Defaults to None.
        The path to the folder to save the resulting specification templates.
        The Tmp folder will be created in case when `output` is not provided.
    overwrite: (bool, optional). Defaults to False.
        Whether to overwrite the result specification YAML if exists.
    merge_config: (bool, optional). Defaults to False.
        Whether to merge the generated specification YAML with the backend configuration.
    ads_config: (str, optional)
        The folder where the ads opctl config located.
    kwargs: (Dict, optional).
        Additional key value arguments.

    Raises
    ------
    ValueError
        If `type` not specified.
    OperatorNotFoundError
        If `operator` not found.
    """
    # validation
    if not type:
        raise ValueError(f"The `type` attribute must be specified.")

    # load operator info
    operator_info: OperatorInfo = OperatorLoader.from_uri(uri=type).load()

    # create TMP folder if one is not provided by user
    if output:
        output = os.path.join(output, "")
        if ads_common_utils.is_path_exists(uri=output) and not overwrite:
            raise ValueError(
                f"The `{output}` already exists, use `--overwrite` option if you wish to overwrite."
            )
    else:
        overwrite = True
        output = operator_utils.create_output_folder(name=type + "/")

    # generating operator specification
    operator_config = {}
    try:
        operator_cmd_module = runpy.run_module(
            f"{operator_info.type}.cmd", run_name="init"
        )
        operator_config = operator_cmd_module.get("init", lambda: "")(
            **{**kwargs, **{"type": type}}
        )

        if not merge_config:
            with fsspec.open(
                os.path.join(output, f"{operator_info.type}.yaml"), mode="w"
            ) as f:
                f.write(yaml.dump(operator_config))
    except Exception as ex:
        logger.warning(
            "The operator's specification was not generated "
            f"because it is not supported by the `{operator_info.type}` operator. "
            "Use --debug option to see the error details."
        )
        logger.debug(ex)
        print_traceback()

    # copy README and original schema files into a destination folder
    for src_file in ("README.md", "schema.yaml", "environment.yaml"):
        ads_common_utils.copy_file(
            uri_src=os.path.join(operator_info.path, src_file),
            uri_dst=output,
            force_overwrite=overwrite,
        )

    # generate supported backend specifications templates YAML
    for key, value in BackendFactory._init_backend_config(
        operator_info=operator_info,
        ads_config=ads_config,
        output=output,
        overwrite=overwrite,
        **kwargs,
    ).items():
        tmp_config = value
        if merge_config and operator_config:
            tmp_config = {**operator_config, OPERATOR_BACKEND_SECTION_NAME: value}

        with fsspec.open(
            os.path.join(
                output,
                f"{operator_info.type}_{'_'.join(key).replace('.','_')}_backend.yaml",
            ),
            mode="w",
        ) as f:
            f.write(yaml.dump(tmp_config))

    logger.info("#" * 50)
    logger.info(f"The auto-generated configs have been placed in: {output}")
    logger.info("#" * 50)


@runtime_dependency(module="docker", install_from=OptionalDependency.OPCTL)
@validate_environment
def build_image(
    type: str = None,
    rebuild_base_image: bool = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Builds the image for the particular operator.

    Parameters
    ----------
    type: (str, optional)
        Type of the operator to build the image.
    rebuild_base_image: (optional, bool)
        If rebuilding both base and operator's images required.
    kwargs: (Dict, optional).
        Additional key value arguments.

    Raises
    ------
    ValueError
        If `type` not specified.
    """
    import docker

    # validation
    if not type:
        raise ValueError(f"The `type` attribute must be specified.")

    # load operator info
    operator_info: OperatorInfo = OperatorLoader.from_uri(uri=type).load()
    logger.info(f"Building Docker image for the `{operator_info.type}` operator.")

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

        result_image_name = operator_utils._build_image(
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
            f"COPY ./operator/ $OPERATOR_DIR/{operator_info.type}/",
            "RUN yum install -y libX11",
        ]
        if os.path.exists(os.path.join(td, "operator", "environment.yaml")):
            run_command.append(
                f"RUN mamba env update -f $OPERATOR_DIR/{operator_info.type}/environment.yaml "
                "--name $CONDA_ENV_NAME && conda clean -afy"
            )

        custom_docker_file = os.path.join(td, "Dockerfile")

        with open(custom_docker_file, "w") as f:
            f.writelines("\n".join(run_command))

        result_image_name = operator_utils._build_image(
            dockerfile=custom_docker_file,
            image_name=operator_info.type,
            tag=operator_info.version,
        )

        logger.info(
            f"The operator image `{result_image_name}` has been successfully built. "
            "To publish the image to OCI Container Registry run the "
            f"`ads operator publish-image -t {operator_info.type}` command"
        )


@runtime_dependency(module="docker", install_from=OptionalDependency.OPCTL)
@validate_environment
def publish_image(
    type: str,
    registry: str = None,
    ads_config: str = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Publishes operator's image to the container registry.

    Parameters
    ----------
    type: (str, optional)
        The operator type to publish image to container registry.
    registry: str
        Container registry.
    ads_config: (str, optional)
        The folder where the ads opctl config located.
    kwargs: (Dict, optional).
        Additional key value arguments.

    Raises
    ------
    ValueError
        If `type` not specified.
    OperatorImageNotFoundError
        If the operator's image doesn't exist.
    """

    import docker

    # validation
    if not type:
        raise ValueError(f"The `type` attribute must be specified.")

    client = docker.from_env()

    # Check if image with given name exists
    image = type
    try:
        client.api.inspect_image(image)
    except docker.errors.ImageNotFound:
        # load operator info
        operator_info: OperatorInfo = OperatorLoader.from_uri(uri=type).load()
        try:
            image = f"{operator_info.type}:{operator_info.version or 'undefined'}"
            # check if the operator's image exists
            client.api.inspect_image(image)
        except docker.errors.ImageNotFound:
            raise OperatorImageNotFoundError(operator_info.type)

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
            f"{operator_info.type}.__main__",
            run_name="verify",
        )
        operator_module.get("verify")(config, **kwargs)
    except InvalidParameterError as ex:
        logger.debug(ex)
        raise ValueError(
            f"The operator's specification is not valid for the `{operator_info.type}` operator. "
            f"{ex}"
        )
    except Exception as ex:
        logger.debug(ex)
        print_traceback()
        raise ValueError(
            f"The validator is not implemented for the `{operator_info.type}` operator."
        )


def build_conda(
    type: str = None,
    conda_pack_folder: str = None,
    overwrite: bool = False,
    ads_config: Union[str, None] = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Builds the conda environment for the particular operator.
    For the service operators, the type needs to be provided.
    For the custom operators, the path (source_folder) to the operator needs to be provided.

    Parameters
    ----------
    type: str
        The type of the operator to build conda environment for.
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
        If `type` not specified.
    """

    # validation
    if not type:
        raise ValueError(f"The `type` attribute must be specified.")

    # load operator info
    operator_info: OperatorInfo = OperatorLoader.from_uri(uri=type).load()
    logger.info(f"Building conda environment for the `{operator_info.type}` operator.")

    # invoke the conda create command
    conda_create(
        name=operator_info.type,
        version=re.sub("[^0-9.]", "", operator_info.version),
        environment_file=os.path.join(operator_info.path, "environment.yaml"),
        conda_pack_folder=conda_pack_folder,
        gpu=operator_info.gpu,
        overwrite=overwrite,
        ads_config=ads_config,
        **kwargs,
    )


def publish_conda(
    type: str = None,
    conda_pack_folder: str = None,
    overwrite: bool = False,
    ads_config: Union[str, None] = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Publishes the conda environment for the particular operator.

    Parameters
    ----------
    type: str
        The type of the operator to generate the specification YAML.
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
        If `type` not specified.
    OperatorCondaNotFoundError
        If the operator's conda environment not exists.
    """

    # validation
    if not type:
        raise ValueError(f"The `type` attribute must be specified.")

    # load operator info
    operator_info: OperatorInfo = OperatorLoader.from_uri(uri=type).load()

    # invoke the conda publish command
    try:
        conda_publish(
            slug=operator_info.conda,
            conda_pack_folder=conda_pack_folder,
            overwrite=overwrite,
            ads_config=ads_config,
            **kwargs,
        )
    except FileNotFoundError:
        raise OperatorCondaNotFoundError(operator_info.type)


def create(
    type: str,
    overwrite: bool = False,
    ads_config: Union[str, None] = None,
    output: str = None,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Creates new operator.

    Parameters
    ----------
    type: str
        The type of the operator to generate the specification YAML.
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


def run(
    config: Dict, backend: Union[Dict, str] = None, **kwargs: Dict[str, Any]
) -> None:
    """
    Runs the operator with the given specification on the targeted backend.

    Parameters
    ----------
    config: Dict
        The operator's config.
    backend: (Union[Dict, str], optional)
        The backend config or backend name to run the operator.
    kwargs: (Dict[str, Any], optional)
        Optional key value arguments to run the operator.
    """
    BackendFactory.backend(
        config=ConfigProcessor(config).step(ConfigMerger, **kwargs),
        backend=backend,
        **kwargs,
    ).run(**kwargs)
