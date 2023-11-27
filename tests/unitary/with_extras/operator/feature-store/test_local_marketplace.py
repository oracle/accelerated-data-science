import io
from unittest.mock import patch, Mock

import mock
import yaml
from kubernetes.client import V1PodList, V1Pod, V1PodStatus, V1PodCondition

from ads.opctl.backend.marketplace.local_marketplace import LocalMarketplaceOperatorBackend
from ads.opctl.backend.marketplace.models.marketplace_type import HelmMarketplaceListingDetails
from ads.opctl.operator.common.operator_loader import OperatorInfo


@patch('ads.opctl.backend.marketplace.local_marketplace.fsspec.open', new_callable=mock.mock_open())
def test_helm_values_to_yaml(file_writer: Mock):
    sample_helm_value = {
        "key1": {
            "inner_key1": "inner_value1",
            "inner_key2": "inner_value2"
        },
        "key2": {
            "inner_key3": "inner_value3",
            "inner_key4": "inner_value4"
        }
    }
    path = LocalMarketplaceOperatorBackend._save_helm_values_to_yaml_(sample_helm_value)
    assert path.endswith('values.yaml')
    file_writer.assert_called_with(path, mode='w')
    file_writer.return_value.__enter__().write.assert_called_once_with(yaml.dump(sample_helm_value))


@patch('ads.opctl.backend.marketplace.local_marketplace.time')
@patch('ads.opctl.backend.marketplace.local_marketplace.client.CoreV1Api')
@patch('ads.opctl.backend.marketplace.local_marketplace.config.load_kube_config')
def test_wait_for_pod_healthy(kube_config: Mock, kube_client: Mock, timer: Mock):
    local_marketplace_operator = LocalMarketplaceOperatorBackend(config={'execution': {}})
    timer.time = Mock(side_effect=[0.0, 11 * 60])
    kube_client.return_value.list_namespaced_pod.return_value.items.__getitem__.return_value.status.conditions.__iter__ = Mock(
        return_value=iter([V1PodCondition(type="Ready", status="True")]))

    assert local_marketplace_operator._wait_for_pod_ready('namespace', 'pod_name') == 0
    kube_client.return_value.list_namespaced_pod.assert_called_once_with(namespace='namespace',
                                                                         label_selector='app.kubernetes.io/instance=pod_name')
    kube_config.assert_called()


@patch('ads.opctl.backend.marketplace.local_marketplace.time')
@patch('ads.opctl.backend.marketplace.local_marketplace.client.CoreV1Api')
@patch('ads.opctl.backend.marketplace.local_marketplace.config.load_kube_config')
def test_wait_for_pod_unhealthy(kube_config: Mock, kube_client: Mock, timer: Mock):
    local_marketplace_operator = LocalMarketplaceOperatorBackend(config={'execution': {}})
    timer.time = Mock(return_value=0.0)
    kube_client.return_value.list_namespaced_pod.return_value.items.__getitem__.return_value.status.conditions.__iter__ = Mock(
        return_value=iter([V1PodCondition(type="Ready", status="False")]))

    assert local_marketplace_operator._wait_for_pod_ready('namespace', 'pod_name') == -1
    kube_client.return_value.list_namespaced_pod.assert_called()
    kube_config.assert_called()


@patch('ads.opctl.backend.marketplace.local_marketplace.export_helm_chart')
@patch('ads.opctl.backend.marketplace.local_marketplace.list_container_images')
def test_export_helm_chart_to_container_registry(list_api: Mock, export_api: Mock):
    pattern = "feature-store-dataplane-api"

    mock_container_summary = Mock()
    mock_container_summary.display_name = f"{pattern}-1"

    list_api.return_value.items.__iter__ = Mock(return_value=iter([mock_container_summary]))
    listing_details = Mock()
    listing_details.container_tag_pattern = [pattern]
    result = LocalMarketplaceOperatorBackend._export_helm_chart_to_container_registry_(listing_details)
    assert pattern in result
    assert result[pattern] == f"{pattern}-1"


@patch('ads.opctl.backend.marketplace.local_marketplace.check_helm_login')
@patch('ads.opctl.backend.marketplace.local_marketplace.check_prerequisites')
@patch('ads.opctl.backend.marketplace.local_marketplace.run_helm_install')
@patch('ads.opctl.backend.marketplace.local_marketplace.MarketplaceBackendRunner')
@patch('ads.opctl.backend.marketplace.local_marketplace.operator_runtime')
def test_run_with_python_success(operator_runtime: Mock, backend_runner: Mock, helm_install_api: Mock,
                                 check_prerequisites: Mock, check_helm_login: Mock
                                 ):
    local_marketplace_operator = LocalMarketplaceOperatorBackend(config={'execution': {}}, operator_info=OperatorInfo())
    local_marketplace_operator._export_helm_chart_to_container_registry_ = Mock()
    local_marketplace_operator._save_helm_values_to_yaml_ = Mock()
    local_marketplace_operator.run_bugfix_command = Mock()
    local_marketplace_operator._wait_for_pod_ready = Mock(return_value=0)
    mock_helm_detail = Mock(spec=HelmMarketplaceListingDetails)
    mock_helm_detail.helm_values = {}
    mock_helm_detail.helm_app_name = 'helm_app_name'
    mock_helm_detail.ocir_fully_qualified_url = 'ocir_fully_qualified_url'
    mock_helm_detail.helm_chart_tag = 'helm_chart_tag'
    mock_helm_detail.namespace = 'namespace'
    backend_runner.return_value.get_listing_details.return_value = mock_helm_detail

    helm_install_api.return_value.returncode = 0
    assert local_marketplace_operator._run_with_python_() == 0


@patch('ads.opctl.backend.marketplace.local_marketplace.check_helm_login')
@patch('ads.opctl.backend.marketplace.local_marketplace.check_prerequisites')
@patch('ads.opctl.backend.marketplace.local_marketplace.run_helm_install')
@patch('ads.opctl.backend.marketplace.local_marketplace.MarketplaceBackendRunner')
@patch('ads.opctl.backend.marketplace.local_marketplace.operator_runtime')
def test_run_with_python_failure(operator_runtime: Mock, backend_runner: Mock, helm_install_api: Mock,
                                 check_prerequisites: Mock,
                                 check_helm_login: Mock
                                 ):
    local_marketplace_operator = LocalMarketplaceOperatorBackend(config={'execution': {}}, operator_info=OperatorInfo())
    local_marketplace_operator._export_helm_chart_to_container_registry_ = Mock()
    local_marketplace_operator._save_helm_values_to_yaml_ = Mock()
    mock_helm_detail = Mock(spec=HelmMarketplaceListingDetails)
    mock_helm_detail.helm_values = {}
    mock_helm_detail.helm_app_name = 'helm_app_name'
    mock_helm_detail.ocir_fully_qualified_url = 'ocir_fully_qualified_url'
    mock_helm_detail.helm_chart_tag = 'helm_chart_tag'
    mock_helm_detail.namespace = 'namespace'
    backend_runner.return_value.get_listing_details.return_value = mock_helm_detail
    helm_install_api.return_value.returncode = -1
    assert local_marketplace_operator._run_with_python_() == -1
