import subprocess
from enum import Enum
from typing import List
import io

import pandas as pd
from ads.common.extended_enum import ExtendedEnumMeta
from ads.opctl import logger


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
    cmd = (
        HelmCommand.Install
        if not _check_if_chart_already_exists_(name, namespace)
        else HelmCommand.Upgrade
    )
    helm_cmd = [
        _HELM_BINARY_,
        cmd,
        name,
        chart,
        *_get_as_flags_(
            namespace=namespace, values=values_yaml_path, version=version, **kwargs
        ),
    ]
    print(" ".join(helm_cmd))
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
        print(stderr)
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
        print(result.stderr)
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
        print(" ".join(helm_cmd))
        print(result.stderr)
        # TODO: Throw proper exception
        raise Exception()
