import io
from unittest.mock import patch, Mock

import mock
import yaml
from kubernetes.client import V1PodList, V1Pod, V1PodStatus, V1PodCondition

from ads.opctl.backend.marketplace.local_marketplace import LocalMarketplaceOperatorBackend


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
    kube_client.return_value.list_namespaced_pod.return_value.items.__getitem__.return_value.status.conditions.__iter__ = Mock(return_value=iter([V1PodCondition(type="Ready", status="True")]))

    assert local_marketplace_operator._wait_for_pod_ready('namespace', 'pod_name') == 0
    kube_client.return_value.list_namespaced_pod.assert_called_once_with(namespace='namespace',
                                                                         label_selector='app.kubernetes.io/instance=pod_name')
    kube_config.assert_called()

@patch('ads.opctl.backend.marketplace.local_marketplace.time')
@patch('ads.opctl.backend.marketplace.local_marketplace.client.CoreV1Api')
@patch('ads.opctl.backend.marketplace.local_marketplace.config.load_kube_config')
def test_wait_for_pod_unhealthy(kube_config: Mock, kube_client: Mock, timer: Mock):
    local_marketplace_operator = LocalMarketplaceOperatorBackend(config={'execution': {}})
    timer.time = Mock(side_effect=[0.0, 11 * 60])
    kube_client.return_value.list_namespaced_pod.return_value.items.__getitem__.return_value.status.conditions.__iter__ = Mock(return_value=iter([V1PodCondition(type="Ready", status="False")]))

    assert local_marketplace_operator._wait_for_pod_ready('namespace', 'pod_name') == -1
    kube_client.return_value.list_namespaced_pod.assert_called_once_with(namespace='namespace',
                                                                         label_selector='app.kubernetes.io/instance=pod_name')
    kube_config.assert_called()