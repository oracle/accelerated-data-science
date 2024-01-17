import base64
from unittest.mock import Mock, patch

import oci.resource_manager.models
import pytest
from typing import List

from ads.opctl.operator.lowcode.feature_store_marketplace.models.apigw_config import (
    APIGatewayConfig,
)

from ads.opctl.operator.lowcode.feature_store_marketplace.const import LISTING_ID
from ads.opctl.operator.lowcode.feature_store_marketplace.models.db_config import (
    DBConfig,
)

from ads.opctl.operator.lowcode.feature_store_marketplace.models.mysql_config import (
    MySqlConfig,
)
from ads.opctl.operator.lowcode.feature_store_marketplace.operator_utils import (
    get_db_details,
    get_latest_listing_version,
    _create_new_stack,
    detect_or_create_stack,
    get_admin_group,
    get_api_gw_details,
)


@patch("click.prompt")
def test_get_db_basic_details(prompt_mock: Mock):
    IP = "10.0.10.122:3306"
    DATABASE_NAME = "featurestore"
    expected_db_config = MySqlConfig()
    expected_db_config.username = "username"
    expected_db_config.auth_type = MySqlConfig.MySQLAuthType.BASIC.value
    expected_db_config.basic_config = MySqlConfig.BasicConfig()
    expected_db_config.basic_config.password = "password"
    expected_db_config.url = (
        f"jdbc:mysql://{IP}/{DATABASE_NAME}?createDatabaseIfNotExist=true"
    )

    prompt_mock.side_effect = [
        expected_db_config.username,
        expected_db_config.auth_type,
        expected_db_config.basic_config.password,
        IP,
        DATABASE_NAME,
    ]
    db_config: DBConfig = get_db_details()

    assert db_config.mysql_config.url == expected_db_config.url
    assert db_config.mysql_config.username == expected_db_config.username
    assert (
        db_config.mysql_config.basic_config.password
        == expected_db_config.basic_config.password
    )


@patch("click.prompt")
def test_get_db_vault_details(prompt_mock: Mock):
    IP = "10.0.10.122:3306"
    DATABASE_NAME = "featurestore"
    expected_db_config = MySqlConfig()
    expected_db_config.username = "username"
    expected_db_config.auth_type = MySqlConfig.MySQLAuthType.VAULT.value
    expected_db_config.vault_config = MySqlConfig.VaultConfig()
    expected_db_config.vault_config.vault_ocid = "vaultocid"
    expected_db_config.vault_config.secret_name = "secretname"
    expected_db_config.url = (
        f"jdbc:mysql://{IP}/{DATABASE_NAME}?createDatabaseIfNotExist=true"
    )
    prompt_mock.side_effect = [
        expected_db_config.username,
        expected_db_config.auth_type,
        expected_db_config.vault_config.vault_ocid,
        expected_db_config.vault_config.secret_name,
        IP,
        DATABASE_NAME,
    ]
    db_config: DBConfig = get_db_details()

    assert db_config.mysql_config.url == expected_db_config.url
    assert db_config.mysql_config.username == expected_db_config.username
    assert (
        db_config.mysql_config.vault_config.vault_ocid
        == expected_db_config.vault_config.vault_ocid
    )
    assert (
        db_config.mysql_config.vault_config.secret_name
        == expected_db_config.vault_config.secret_name
    )


@patch(
    "ads.opctl.operator.lowcode.feature_store_marketplace.operator_utils.OCIClientFactory"
)
@patch("ads.common.auth.default_signer")
def test_get_latest_listing_revision(auth_mock: Mock, client_factory: Mock):
    client_mock = Mock()
    client_factory.return_value = Mock(create_client=Mock(return_value=client_mock))
    get_latest_listing_version("compartment_id")
    client_mock.get_listing.assert_called_once_with(
        LISTING_ID, compartment_id="compartment_id"
    )


@patch(
    "ads.opctl.operator.lowcode.feature_store_marketplace.operator_utils.OCIClientFactory"
)
@patch("ads.common.auth.default_signer")
def test_get_latest_listing_revision_with_exception(
    auth_mock: Mock, client_factory: Mock
):
    class TestException(Exception):
        pass

    def throw_exception(*args, **kwargs):
        raise TestException()

    client_mock = Mock()

    client_factory.return_value = Mock(create_client=Mock(return_value=client_mock))
    client_mock.get_listing = Mock()
    client_mock.get_listing.side_effect = throw_exception
    with pytest.raises(TestException):
        get_latest_listing_version("compartment_id")


@patch(
    "ads.opctl.operator.lowcode.feature_store_marketplace.operator_utils.OCIClientFactory"
)
@patch("ads.common.auth.default_signer")
@patch("ads.opctl.operator.lowcode.feature_store_marketplace.operator_utils.requests")
def test_create_new_stack(requests_mock: Mock, auth_mock: Mock, client_factory: Mock):
    zip_file = b"hello"
    requests_mock.get = Mock()
    requests_mock.get.return_value = Mock(content=zip_file)

    def validate_stack_details(
        stack_details: oci.resource_manager.models.CreateStackDetails,
    ):
        import oci.resource_manager.models as models

        source_details: models.CreateZipUploadConfigSourceDetails = (
            stack_details.config_source
        )
        assert (
            source_details.zip_file_base64_encoded
            == base64.b64encode(zip_file).decode()
        )
        stack = oci.resource_manager.models.Stack()
        stack.id = "ID"
        return oci.Response(data=stack, request=None, headers=None, status=None)

    client_mock = Mock()
    client_factory.return_value = Mock(create_client=Mock(return_value=client_mock))
    client_mock.create_stack = Mock()
    client_mock.create_stack.side_effect = validate_stack_details
    _create_new_stack(APIGatewayConfig())
    assert client_mock.create_stack.call_count == 1


@patch(
    "ads.opctl.operator.lowcode.feature_store_marketplace.operator_utils.OCIClientFactory"
)
@patch("ads.common.auth.default_signer")
@patch(
    "ads.opctl.operator.lowcode.feature_store_marketplace.operator_utils._create_new_stack"
)
@patch(
    "ads.opctl.operator.lowcode.feature_store_marketplace.operator_utils._print_stack_detail"
)
@patch("click.prompt")
def test_detect_or_create_new_stack(
    mock_prompt: Mock,
    print_mock: Mock,
    create_new_mock_stack: Mock,
    auth_mock: Mock,
    client_factory: Mock,
):
    import oci.resource_manager.models as models

    ocid = "id"
    mock_prompt.side_effect = ["1"]
    client_mock = Mock()
    client_factory.return_value = Mock(create_client=Mock(return_value=client_mock))
    client_mock.list_stacks = Mock()
    create_new_mock_stack.return_value = ocid
    stacks: List[models.StackSummary] = [models.StackSummary(), models.StackSummary()]
    client_mock.list_stacks.return_value = oci.Response(
        data=stacks, request=None, headers=None, status=None
    )
    assert detect_or_create_stack(apigw_config=APIGatewayConfig()) == ocid
    assert print_mock.call_count == len(stacks)
    assert create_new_mock_stack.call_count == 1


@patch(
    "ads.opctl.operator.lowcode.feature_store_marketplace.operator_utils.OCIClientFactory"
)
@patch("ads.common.auth.default_signer")
@patch(
    "ads.opctl.operator.lowcode.feature_store_marketplace.operator_utils._print_stack_detail"
)
@patch("click.prompt")
def test_detect_or_create_existing_stack(
    mock_prompt: Mock,
    print_mock: Mock,
    auth_mock: Mock,
    client_factory: Mock,
):
    ocid = "id"
    import oci.resource_manager.models as models

    mock_prompt.side_effect = ["2", ocid]
    client_mock = Mock()
    client_factory.return_value = Mock(create_client=Mock(return_value=client_mock))
    client_mock.list_stacks = Mock()

    stacks: List[models.StackSummary] = []
    client_mock.list_stacks.return_value = oci.Response(
        data=stacks, request=None, headers=None, status=None
    )
    assert detect_or_create_stack(apigw_config=APIGatewayConfig()) == ocid
    assert print_mock.call_count == len(stacks)


@patch(
    "ads.opctl.operator.lowcode.feature_store_marketplace.operator_utils.OCIClientFactory"
)
@patch("ads.common.auth.default_signer")
def test_get_admin_group(auth_mock: Mock, client_factory: Mock):
    import oci.identity.models as models

    ocid = "id"
    client_mock = Mock()
    client_factory.return_value = Mock(create_client=Mock(return_value=client_mock))
    client_mock.list_groups = Mock()
    groups: List[models.Group] = [models.Group(), models.Group()]
    groups[0].id = ocid
    client_mock.list_groups.return_value = oci.Response(
        data=groups, request=None, headers=None, status=None
    )
    assert get_admin_group("tenant_id") == ocid
    assert client_mock.list_groups.call_count == 1


@patch(
    "ads.opctl.operator.lowcode.feature_store_marketplace.operator_utils.OCIClientFactory"
)
@patch("ads.common.auth.default_signer")
def test_get_admin_group_with_no_groups(auth_mock: Mock, client_factory: Mock):
    import oci.identity.models as models

    ocid = "id"
    client_mock = Mock()
    client_factory.return_value = Mock(create_client=Mock(return_value=client_mock))
    client_mock.list_groups = Mock()
    groups: List[models.Group] = []
    client_mock.list_groups.return_value = oci.Response(
        data=groups, request=None, headers=None, status=None
    )
    assert get_admin_group("tenant_id") is None
    assert client_mock.list_groups.call_count == 1


@patch("click.prompt")
@patch("click.confirm")
@patch("ads.opctl.operator.lowcode.feature_store_marketplace.operator_utils.get_region")
@patch(
    "ads.opctl.operator.lowcode.feature_store_marketplace.operator_utils.detect_or_create_stack"
)
@patch(
    "ads.opctl.operator.lowcode.feature_store_marketplace.operator_utils.get_admin_group"
)
def test_get_api_gw_details(
    mock_get_admin_group: Mock,
    mock_detect_or_create_stack: Mock,
    mock_region: Mock,
    mock_confirm: Mock,
    mock_prompt: Mock,
):
    admin_group = "group"
    comp_id = "comp_id"
    stack_id = "stack_id"
    region = "ashburn"
    mock_confirm.side_effect = ["Y"]
    mock_region.return_value = None
    mock_prompt.side_effect = [comp_id, region, admin_group]
    mock_get_admin_group.return_value = ""
    mock_detect_or_create_stack.return_value = stack_id
    api_gw_details: APIGatewayConfig = get_api_gw_details("")
    assert mock_detect_or_create_stack.call_count == 1
    assert mock_confirm.call_count == 1
    assert mock_prompt.call_count == 3
    assert api_gw_details.root_compartment_id == comp_id
    assert api_gw_details.stack_id == stack_id
    assert api_gw_details.region == region


@patch("click.prompt")
@patch("click.confirm")
@patch("ads.opctl.operator.lowcode.feature_store_marketplace.operator_utils.get_region")
@patch(
    "ads.opctl.operator.lowcode.feature_store_marketplace.operator_utils.detect_or_create_stack"
)
@patch(
    "ads.opctl.operator.lowcode.feature_store_marketplace.operator_utils.get_admin_group"
)
def test_get_api_gw_details_auto_detect_compartment_and_region(
    mock_get_admin_group: Mock,
    mock_detect_or_create_stack: Mock,
    mock_region: Mock,
    mock_confirm: Mock,
    mock_prompt: Mock,
):
    admin_group = "group"
    comp_id = "tenancy"
    stack_id = "stack_id"
    region = "ashburn"
    mock_confirm.side_effect = ["Y"]
    mock_region.return_value = region
    mock_prompt.side_effect = [admin_group]
    mock_get_admin_group.return_value = ""
    mock_detect_or_create_stack.return_value = stack_id
    api_gw_details: APIGatewayConfig = get_api_gw_details(comp_id)
    assert mock_detect_or_create_stack.call_count == 1
    assert mock_confirm.call_count == 1
    assert mock_prompt.call_count == 1
    assert api_gw_details.root_compartment_id == comp_id
    assert api_gw_details.stack_id == stack_id
    assert api_gw_details.region == region
