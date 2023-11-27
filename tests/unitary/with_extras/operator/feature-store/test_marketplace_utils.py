import os
from unittest.mock import Mock, patch

from ads.opctl.backend.marketplace.marketplace_utils import set_kubernetes_session_token_env, get_docker_bearer_token, \
    _export_helm_chart_, list_container_images, export_helm_chart_to_container_registry


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
    token_client.call_api.assert_called_once_with(resource_path='/docker/token', method='GET',
                                                  response_type='SecurityToken')


@patch('ads.opctl.backend.marketplace.marketplace_utils.get_marketplace_client')
@patch('ads.opctl.backend.marketplace.marketplace_utils.oci')
def test_export_helm_chart_success(oci: Mock, marketplace_client: Mock):
    oci.wait_until.return_value.data.status = "SUCCESS"
    listing_details = Mock()
    _export_helm_chart_(listing_details)
    marketplace_client.return_value.export_listing.assert_called_once()
    oci.wait_until.assert_called_once()


@patch('ads.opctl.backend.marketplace.marketplace_utils.OCIClientFactory')
def test_list_container_image(oci_factory: Mock):
    list_container_images(compartment_id="compartment_id", ocir_image_path="ocir_image_path")
    oci_factory.return_value.artifacts.list_container_images.assert_called_once()


@patch('ads.opctl.backend.marketplace.marketplace_utils._export_helm_chart_')
@patch('ads.opctl.backend.marketplace.marketplace_utils.list_container_images')
def test_export_helm_chart_to_container_registry(list_api: Mock, export_api: Mock):
    pattern = "feature-store-dataplane-api"

    mock_container_summary = Mock()
    mock_container_summary.display_name = f"{pattern}-1"

    list_api.return_value.items.__iter__ = Mock(return_value=iter([mock_container_summary]))
    listing_details = Mock()
    listing_details.container_tag_pattern = [pattern]
    result = export_helm_chart_to_container_registry(listing_details)
    assert pattern in result
    assert result[pattern] == f"{pattern}-1"
