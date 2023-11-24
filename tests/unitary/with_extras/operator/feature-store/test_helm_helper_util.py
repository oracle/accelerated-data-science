import subprocess

import pandas as pd
import pytest

from ads.opctl.backend.marketplace.helm_helper import run_helm_install, _HELM_BINARY_, HelmCommand, _get_as_flags_, \
    run_helm_list, run_helm_login, _check_if_chart_already_exists_, check_helm_pull, HelmPullStatus
from unittest.mock import patch, Mock, create_autospec

name = "NAME"
chart = "CHART_NAME"
version = "VERSION"
namespace = "NAMESPACE"
values_yaml_path = "path/to/values.yaml"
kwargs = {"key": "value"}


@patch("ads.opctl.backend.marketplace.helm_helper.subprocess.run")
@patch("ads.opctl.backend.marketplace.helm_helper._check_if_chart_already_exists_")
def test_helm_install(mock_check_chart_exist: Mock, subprocess_mock: Mock):
    mock_check_chart_exist.return_value = False
    run_helm_install(name, chart, version, namespace, values_yaml_path, **kwargs)
    mock_check_chart_exist.assert_called_with(name, namespace)

    helm_cmd = [
        _HELM_BINARY_,
        HelmCommand.Install,
        name,
        chart,
        *_get_as_flags_(
            namespace=namespace, values=values_yaml_path, version=version, **kwargs
        ),
    ]

    subprocess_mock.assert_called_with(helm_cmd)


@patch("ads.opctl.backend.marketplace.helm_helper.subprocess.run")
@patch("ads.opctl.backend.marketplace.helm_helper._check_if_chart_already_exists_")
def test_helm_upgrade(mock_check_chart_exist: Mock, subprocess_mock: Mock):
    mock_check_chart_exist.return_value = True
    run_helm_install(name, chart, version, namespace, values_yaml_path, **kwargs)
    mock_check_chart_exist.assert_called_with(name, namespace)
    helm_cmd = [
        _HELM_BINARY_,
        HelmCommand.Upgrade,
        name,
        chart,
        *_get_as_flags_(
            namespace=namespace, values=values_yaml_path, version=version, **kwargs
        ),
    ]
    subprocess_mock.assert_called_with(helm_cmd)


@patch("ads.opctl.backend.marketplace.helm_helper.subprocess.run")
def test_helm_list_returns_single_value(subprocess_mock: Mock):
    header = ["NAME", "NAMESPACE", "REVISION", "UPDATED", "STATUS", "CHART", "APP VERSION"]
    data = ["fs-dp-api-test", "feature-store", "4", "2023-11-22 10:22:13.425579296 +0530 IST", "deployed",
            "feature-store-dp-api-1.0", "0.1.270.marketplace-vuls"]
    std_out = "\t".join(header) + "\n" + "\t".join(data)

    list_result = subprocess.CompletedProcess(args="", returncode=0, stdout=std_out.encode())
    subprocess_mock.return_value = list_result
    result = run_helm_list(namespace, **kwargs)
    assert len(result) == 1
    assert "NAME" in result.columns
    assert any(result['NAME'] == "fs-dp-api-test")


@patch("ads.opctl.backend.marketplace.helm_helper.subprocess.run")
def test_helm_list_returns_no_value(subprocess_mock: Mock):
    header = ["NAME", "NAMESPACE", "REVISION", "UPDATED", "STATUS", "CHART", "APP VERSION"]
    std_out = "\t".join(header)

    list_result = subprocess.CompletedProcess(args="", returncode=0, stdout=std_out.encode())
    subprocess_mock.return_value = list_result
    result = run_helm_list(namespace, **kwargs)
    assert len(result) == 0
    assert "NAME" in result.columns


@patch("ads.opctl.backend.marketplace.helm_helper.subprocess.run")
def test_helm_list_throws_exception(subprocess_mock: Mock):
    list_result = subprocess.CompletedProcess(args="", returncode=1, stderr="Some Exception")
    subprocess_mock.return_value = list_result
    with pytest.raises(Exception) as e_info:
        run_helm_list(namespace, **kwargs)


@patch("ads.opctl.backend.marketplace.helm_helper.subprocess.run")
def test_helm_login_success(subprocess_mock: Mock):
    subprocess_return = subprocess.CompletedProcess(args="", returncode=0)
    subprocess_mock.return_value = subprocess_return
    run_helm_login('oci_repo', 'token')
    subprocess_mock.assert_called_with(
        ['helm', 'registry', 'login', 'oci_repo', '--username', 'BEARER_TOKEN', '--password', 'token'],
        capture_output=True)


@patch("ads.opctl.backend.marketplace.helm_helper.subprocess.run")
def test_helm_login_throws_exception(subprocess_mock: Mock):
    subprocess_return = subprocess.CompletedProcess(args="", returncode=1, stderr="Some Exception")
    subprocess_mock.return_value = subprocess_return
    with pytest.raises(Exception) as e_info:
        run_helm_login('oci_repo', 'token')
    subprocess_mock.assert_called_with(
        ['helm', 'registry', 'login', 'oci_repo', '--username', 'BEARER_TOKEN', '--password', 'token'],
        capture_output=True)


@patch("ads.opctl.backend.marketplace.helm_helper.subprocess.run")
def test_helm_pull_success(subprocess_mock: Mock):
    subprocess_return = subprocess.CompletedProcess(args="", returncode=0, stderr=b"")
    subprocess_mock.return_value = subprocess_return
    assert check_helm_pull('helm_chart_url', 'version') == HelmPullStatus.SUCCESS
    subprocess_mock.assert_called_with(
        ['helm', 'pull', 'helm_chart_url', '--version', 'version'],
        capture_output=True)


@patch("ads.opctl.backend.marketplace.helm_helper.subprocess.run")
def test_helm_pull_unauthorized(subprocess_mock: Mock):
    subprocess_return = subprocess.CompletedProcess(args="", returncode=1, stderr=b"unauthorized")
    subprocess_mock.return_value = subprocess_return
    assert check_helm_pull('helm_chart_url', 'version') == HelmPullStatus.AUTHENTICATION_FAILURE
    subprocess_mock.assert_called_with(
        ['helm', 'pull', 'helm_chart_url', '--version', 'version'],
        capture_output=True)


@patch("ads.opctl.backend.marketplace.helm_helper.subprocess.run")
def test_helm_pull_unknown_failure(subprocess_mock: Mock):
    subprocess_return = subprocess.CompletedProcess(args="", returncode=1, stderr=b"some failure")
    subprocess_mock.return_value = subprocess_return
    assert check_helm_pull('helm_chart_url', 'version') == HelmPullStatus.UNKNOWN_FAILURE
    subprocess_mock.assert_called_with(
        ['helm', 'pull', 'helm_chart_url', '--version', 'version'],
        capture_output=True)


def test_get_as_flags_():
    args = {
        "key1": "value1",
        "key2": "value2"
    }
    assert _get_as_flags_(**args) == ["--key1", "value1", "--key2", "value2"]


@patch("ads.opctl.backend.marketplace.helm_helper.run_helm_list")
def test_check_helm_chart_exist_when_chart_do_exist(helm_list_cmd: Mock):
    header = ["NAME", "NAMESPACE", "REVISION", "UPDATED", "STATUS", "CHART", "APP VERSION"]
    data = ["fs-dp-api-test", "feature-store", "4", "2023-11-22 10:22:13.425579296 +0530 IST", "deployed",
            "feature-store-dp-api-1.0", "0.1.270.marketplace-vuls"]
    helm_list_cmd.return_value = pd.DataFrame([data], columns=header)
    assert _check_if_chart_already_exists_("fs-dp-api-test", namespace)


@patch("ads.opctl.backend.marketplace.helm_helper.run_helm_list")
def test_check_helm_chart_exist_when_chart_do_not_exist(helm_list_cmd: Mock):
    header = ["NAME", "NAMESPACE", "REVISION", "UPDATED", "STATUS", "CHART", "APP VERSION"]
    data = ["some-other-chart", "feature-store", "4", "2023-11-22 10:22:13.425579296 +0530 IST", "deployed",
            "feature-store-dp-api-1.0", "0.1.270.marketplace-vuls"]
    helm_list_cmd.return_value = pd.DataFrame([data], columns=header)
    assert not _check_if_chart_already_exists_("fs-dp-api-test", namespace)


@patch("ads.opctl.backend.marketplace.helm_helper.run_helm_list")
def test_check_helm_chart_exist_when_list_is_empty(helm_list_cmd: Mock):
    helm_list_cmd.return_value = pd.DataFrame()
    assert not _check_if_chart_already_exists_("fs-dp-api-test", namespace)
