import click
from ads.opctl import logger

from ads.opctl.backend.marketplace.marketplace_utils import Color, print_heading
from ads.opctl.operator.lowcode.feature_store_marketplace.models.mysql_config import (
    MySqlConfig,
)

from ads.opctl.operator.lowcode.feature_store_marketplace.models.db_config import (
    DBConfig,
)


def get_db_details() -> DBConfig:
    jdbc_url = "jdbc://mysql://{}/{}?createDatabaseIfNotExist=true"
    mysql_db_config = MySqlConfig()
    print_heading(
        f"MySQL database configuration",
        colors=[Color.BOLD, Color.BLUE],
        prefix_newline_count=2,
    )
    mysql_db_config.username = click.prompt("Username", default="admin")
    mysql_db_config.auth_type = MySqlConfig.MySQLAuthType(
        click.prompt(
            "Is password provided as plain-text or via a Vault secret?\n"
            "(https://docs.oracle.com/en-us/iaas/Content/KeyManagement/Concepts/keyoverview.htm)",
            type=click.Choice(MySqlConfig.MySQLAuthType.values()),
            default=MySqlConfig.MySQLAuthType.BASIC.value,
        )
    )
    if mysql_db_config.auth_type == MySqlConfig.MySQLAuthType.BASIC:
        basic_auth_config = MySqlConfig.BasicConfig()
        basic_auth_config.password = click.prompt(f"Password", hide_input=True)
        mysql_db_config.basic_config = basic_auth_config

    elif mysql_db_config.auth_type == MySqlConfig.MySQLAuthType.VAULT:
        vault_auth_config = MySqlConfig.VaultConfig()
        vault_auth_config.vault_ocid = click.prompt("Vault OCID")
        vault_auth_config.secret_name = click.prompt(
            "Name of the secret having password"
        )
        mysql_db_config.vault_config = vault_auth_config

    mysql_jdbc_ip = click.prompt(
        "IP address using which the database can be access inside the Kubernetes cluster"
        " (example: 10.0.0.1:3306)"
    )
    db_name = click.prompt(
        "Database name (will be auto created if it doesn't already exist)",
        default="FeatureStore",
    )
    mysql_db_config.url = jdbc_url.format(mysql_jdbc_ip, db_name)
    logger.debug(f"MySQL jdbc url generated is: {mysql_db_config.url}")
    db_config = DBConfig()
    db_config.mysql_config = mysql_db_config
    return db_config
