import os
from unittest.mock import Mock, patch

from kubernetes.client import V1PodCondition

from ads.opctl.backend.marketplace.marketplace_utils import set_kubernetes_session_token_env, get_docker_bearer_token, \
    export_helm_chart, list_container_images, wait_for_pod_ready


def test_set_kubernetes_session_token_env():
    profile_name = "PROFILE_NAME"
    set_kubernetes_session_token_env(profile_name)

    assert os.getenv("OCI_CLI_AUTH") == "security_token"
    assert os.getenv("OCI_CLI_PROFILE") == profile_name


@patch('ads.opctl.backend.marketplace.marketplace_utils.OCIClientFactory')
def test_get_docker_bearer_token(client_factory: Mock):
    mock_token = '{"token":"TOKEN"}'
    token_client = Mock()
    token_client.call_api.return_value.data = mock_token
    client_factory.return_value.create_client.return_value = token_client
    ocir_repo = "iad.ocir.io/idogsu2ylimg/feature-store-data-plane-api-helidon/"
    assert get_docker_bearer_token(ocir_repo) == mock_token
    token_client.call_api.assert_called_once_with(resource_path='/docker/token', method='GET', response_type='SecurityToken')


@patch('ads.opctl.backend.marketplace.marketplace_utils.get_marketplace_client')
@patch('ads.opctl.backend.marketplace.marketplace_utils.oci')
def test_export_helm_chart_success(oci: Mock, marketplace_client: Mock):
    oci.wait_until.return_value.data.status = "SUCCESS"
    listing_details = Mock()
    export_helm_chart(listing_details)
    marketplace_client.return_value.export_listing.assert_called_once()
    oci.wait_until.assert_called_once()

@patch('ads.opctl.backend.marketplace.marketplace_utils.OCIClientFactory')
def test_list_container_image(oci_factory: Mock):
    listing_details = Mock()
    list_container_images(listing_details)
    oci_factory.return_value.artifacts.list_container_images.assert_called_once()

@patch('ads.opctl.backend.marketplace.marketplace_utils.time')
@patch('kubernetes.client.CoreV1Api')
@patch('kubernetes.config.load_kube_config')
def test_wait_for_pod_unhealthy(kube_config: Mock, kube_client: Mock, timer: Mock):
    timer.time = Mock(return_value=0.0)
    kube_client.return_value.list_namespaced_pod.return_value.items.__getitem__.return_value.status.conditions.__iter__ = Mock(
        return_value=iter([V1PodCondition(type="Ready", status="False")]))

    assert wait_for_pod_ready('namespace', 'pod_name') == -1
    kube_client.return_value.list_namespaced_pod.assert_called()
    kube_config.assert_called()

@patch('ads.opctl.backend.marketplace.marketplace_utils.time')
@patch('kubernetes.client.CoreV1Api')
@patch('kubernetes.config.load_kube_config')
def test_wait_for_pod_healthy(kube_config: Mock, kube_client: Mock, timer: Mock):
    timer.time = Mock(side_effect=[0.0, 11 * 60])
    kube_client.return_value.list_namespaced_pod.return_value.items.__getitem__.return_value.status.conditions.__iter__ = Mock(
        return_value=iter([V1PodCondition(type="Ready", status="True")]))

    assert wait_for_pod_ready('namespace', 'pod_name') == 0
    kube_client.return_value.list_namespaced_pod.assert_called_once_with(namespace='namespace',
                                                                         label_selector='app.kubernetes.io/instance=pod_name')
    kube_config.assert_called()


