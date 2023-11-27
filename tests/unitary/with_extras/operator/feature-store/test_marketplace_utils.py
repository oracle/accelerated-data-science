import os
from unittest.mock import Mock, patch

from ads.opctl.backend.marketplace.marketplace_utils import set_kubernetes_session_token_env, get_docker_bearer_token, \
    _export_helm_chart_, list_container_images


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
    _export_helm_chart_(listing_details)
    marketplace_client.return_value.export_listing.assert_called_once()
    oci.wait_until.assert_called_once()

@patch('ads.opctl.backend.marketplace.marketplace_utils.OCIClientFactory')
def test_list_container_image(oci_factory: Mock):
    listing_details = Mock()
    list_container_images(listing_details)
    oci_factory.return_value.artifacts.list_container_images.assert_called_once()
