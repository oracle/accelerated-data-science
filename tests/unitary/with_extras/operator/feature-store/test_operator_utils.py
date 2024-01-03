from unittest.mock import Mock, patch

import pytest

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
