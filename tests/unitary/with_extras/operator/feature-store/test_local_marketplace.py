import io
from unittest.mock import patch, Mock

import mock
import yaml

from ads.opctl.backend.marketplace.local_marketplace import (
    LocalMarketplaceOperatorBackend,
)
from ads.opctl.backend.marketplace.models.marketplace_type import (
    HelmMarketplaceListingDetails,
)
from ads.opctl.operator.common.operator_loader import OperatorInfo


@patch(
    "ads.opctl.backend.marketplace.local_marketplace.fsspec.open",
    new_callable=mock.mock_open(),
)
def test_helm_values_to_yaml(file_writer: Mock):
    sample_helm_value = {
        "key1": {"inner_key1": "inner_value1", "inner_key2": "inner_value2"},
        "key2": {"inner_key3": "inner_value3", "inner_key4": "inner_value4"},
    }
    path = LocalMarketplaceOperatorBackend._save_helm_values_to_yaml_(sample_helm_value)
    assert path.endswith("values.yaml")
    file_writer.assert_called_with(path, mode="w")
    file_writer.return_value.__enter__().write.assert_called_once_with(
        yaml.dump(sample_helm_value)
    )


@patch("ads.opctl.backend.marketplace.local_marketplace.MarketplaceBackendRunner")
@patch("ads.opctl.backend.marketplace.local_marketplace.operator_runtime")
def test_run_with_python_success(operator_runtime: Mock, backend_runner: Mock):
    local_marketplace_operator = LocalMarketplaceOperatorBackend(
        config={"execution": {}}, operator_info=OperatorInfo()
    )
    local_marketplace_operator.process_helm_listing = Mock(return_value=0)

    mock_helm_detail = Mock(spec=HelmMarketplaceListingDetails)
    mock_helm_detail.helm_values = {}
    mock_helm_detail.helm_app_name = "helm_app_name"
    mock_helm_detail.ocir_fully_qualified_url = "ocir_fully_qualified_url"
    mock_helm_detail.helm_chart_tag = "helm_chart_tag"
    mock_helm_detail.namespace = "namespace"
    backend_runner.return_value.get_listing_details.return_value = mock_helm_detail
    # helm_install_api.return_value.returncode = 0
    assert local_marketplace_operator._run_with_python_() == 0


@patch("ads.opctl.backend.marketplace.local_marketplace.check_helm_login")
@patch("ads.opctl.backend.marketplace.local_marketplace.check_prerequisites")
@patch("ads.opctl.backend.marketplace.local_marketplace.run_helm_install")
@patch("ads.opctl.backend.marketplace.local_marketplace.MarketplaceBackendRunner")
@patch("ads.opctl.backend.marketplace.local_marketplace.operator_runtime")
def test_run_with_python_failure(
    operator_runtime: Mock,
    backend_runner: Mock,
    helm_install_api: Mock,
    check_prerequisites: Mock,
    check_helm_login: Mock,
):
    local_marketplace_operator = LocalMarketplaceOperatorBackend(
        config={"execution": {}}, operator_info=OperatorInfo()
    )
    local_marketplace_operator.process_helm_listing = Mock(return_value=-1)
    mock_helm_detail = Mock(spec=HelmMarketplaceListingDetails)
    mock_helm_detail.helm_values = {}
    mock_helm_detail.helm_app_name = "helm_app_name"
    mock_helm_detail.ocir_fully_qualified_url = "ocir_fully_qualified_url"
    mock_helm_detail.helm_chart_tag = "helm_chart_tag"
    mock_helm_detail.namespace = "namespace"
    backend_runner.return_value.get_listing_details.return_value = mock_helm_detail
    helm_install_api.return_value.returncode = -1
    assert local_marketplace_operator._run_with_python_() == -1
