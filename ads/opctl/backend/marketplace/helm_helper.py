#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import subprocess
from enum import Enum
from typing import List
import io

import click
import pandas as pd
from ads.opctl.backend.marketplace.models.marketplace_type import (
    HelmMarketplaceListingDetails,
)

from ads.common.extended_enum import ExtendedEnumMeta
from ads.opctl import logger
from ads.opctl.backend.marketplace.marketplace_utils import (
    StatusIcons,
    get_docker_bearer_token,
    WARNING,
    Color,
)


class HelmCommand(str, metaclass=ExtendedEnumMeta):
    """Supported Helm commands."""

    Install = "install"
    Upgrade = "upgrade"
    List = "list"
    Pull = "pull"
    Registry = "registry"


class HelmPullStatus(Enum):
    SUCCESS = "success"
    UNKNOWN_FAILURE = ("unknown_failure",)
    AUTHENTICATION_FAILURE = "authentication_failure"


_HELM_BINARY_ = "helm"


def run_helm_install(
    name: str, chart: str, version: str, namespace: str, values_yaml_path: str, **kwargs
) -> subprocess.CompletedProcess:
    helm_cmd = [
        _HELM_BINARY_,
        HelmCommand.Upgrade,
        name,
        chart,
        *_get_as_flags_(
            namespace=namespace,
            values=values_yaml_path,
            version=version,
            timeout="300s",
            **kwargs,
        ),
        "--wait",
        "-i",
    ]
    print(f"\n{Color.BLUE}{' '.join(helm_cmd)}{Color.END}")
    return subprocess.run(helm_cmd)


def _get_as_flags_(**kwargs) -> List[str]:
    flags = []
    for key, value in kwargs.items():
        flags.extend([f"--{key}", value])
    return flags


def _check_if_chart_already_exists_(name: str, namespace: str) -> bool:
    logger.debug(f"Checking if chart `{name}` already exists in namespace={namespace}")
    helm_list: pd.DataFrame = run_helm_list(namespace=namespace)
    for chart in helm_list.get("NAME", []):
        if chart == name:
            return True
    return False


def check_helm_login(listing_details: HelmMarketplaceListingDetails):
    print(f"{Color.BLUE}Checking if Helm client is authenticated{Color.END}")
    status = check_helm_pull(
        helm_chart_url=listing_details.helm_fully_qualified_url,
        version=listing_details.helm_chart_tag,
    )
    if status == HelmPullStatus.UNKNOWN_FAILURE:
        # Todo throw correct exception
        raise Exception
    elif status == HelmPullStatus.SUCCESS:
        print(f"Helm client is authenticated {StatusIcons.CHECK}")
        return
    elif status == HelmPullStatus.AUTHENTICATION_FAILURE:
        response = click.confirm(
            text=f"{WARNING} {Color.RED}Helm is unable to access OCIR due to authentication failure.{Color.END}\nDo you want to allow operator to automatically try to fix the issue by setting up bearer token authentication?",
            abort=True,
            default=True,
        )
        if response:
            token = get_docker_bearer_token(listing_details.ocir_registry)
            run_helm_login(ocir_repo=listing_details.ocir_registry, token=token.token)
            status = check_helm_pull(
                helm_chart_url=listing_details.helm_fully_qualified_url,
                version=listing_details.helm_chart_tag,
            )
            status = HelmPullStatus.SUCCESS
            if status != HelmPullStatus.SUCCESS:
                print(f"Unable to setup helm authentication. {StatusIcons.CROSS}")
                # Todo throw correct exception
                raise Exception
            else:
                print(f"Successfully setup authentication for Helm {StatusIcons.CHECK}")


def check_helm_pull(helm_chart_url: str, version: str) -> HelmPullStatus:
    logger.debug(f"Checking if chart `{helm_chart_url}` is accessible")
    helm_cmd = [
        _HELM_BINARY_,
        HelmCommand.Pull,
        helm_chart_url,
        *_get_as_flags_(version=f"{version}"),
    ]
    logger.debug(" ".join(helm_cmd))
    result = subprocess.run(helm_cmd, capture_output=True)
    stderr = result.stderr.decode("utf-8")
    if result.returncode == 0:
        return HelmPullStatus.SUCCESS
    elif "unauthorized" in stderr.lower():
        logger.debug(stderr)
        return HelmPullStatus.AUTHENTICATION_FAILURE
    else:
        logger.error(stderr)
        return HelmPullStatus.UNKNOWN_FAILURE


def run_helm_login(ocir_repo: str, token: str):
    helm_cmd = [
        _HELM_BINARY_,
        HelmCommand.Registry,
        "login",
        ocir_repo,
        *_get_as_flags_(username="BEARER_TOKEN", password=token),
    ]
    logger.debug(" ".join(helm_cmd[:-1]))
    result = subprocess.run(helm_cmd, capture_output=True)
    if result.returncode == 0:
        pass
    else:
        logger.error(result.stderr)
        # TODO: Throw proper exception
        raise Exception()


def run_helm_list(namespace: str, **kwargs) -> pd.DataFrame:
    helm_cmd = [
        _HELM_BINARY_,
        HelmCommand.List,
        *_get_as_flags_(namespace=namespace, **kwargs),
    ]
    logger.debug(" ".join(helm_cmd))
    result = subprocess.run(helm_cmd, capture_output=True)
    if result.returncode == 0:
        return pd.read_csv(
            io.BytesIO(result.stdout), delimiter=r"\s*\t\s*", engine="python"
        )
    else:
        print(f"\n{Color.BLUE}{' '.join(helm_cmd)}{Color.END}")
        print(result.stderr)
        # TODO: Throw proper exception
        raise Exception()
