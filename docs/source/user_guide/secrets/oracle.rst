Oracle Database Connection without a Wallet File
================================================

To connect to an Oracle Database you need the following:

- user name
- password
- hostname
- service name or sid
- port. Default is 1521

The ``OracleDBSecretKeeper`` class saves the Oracle Database credentials to the OCI Vault service.


Saving Credentials
------------------

**Prerequisites**

- OCID of the vault created in the OCI Console.
- OCID of the master key to use for encrypting the secret content stored inside vault.
- OCID of the compartment where the vaut resides. This defaults to the compartment of the notebook session when
  used in a Data Science notebook session.

**OracleDBSecretKeeper**

OracleDBSecretKeeper uses following parameter:

- ``user_name: str``. The user name to be stored.
- ``password: str``. The password of the database.
- ``service_name: (str, optional)``. The service name of the database.
- ``sid: (str, optional)``. The SID of the database if the service name is not available.
- ``host: str``. The hostname of the database.
- ``port: (str, optional). Default 1521``. Port number of the database service.
- ``dsn: (str, optional)``. The DSN string if available.
- ``vault_id: str``. OCID of the vault.
- ``key_id: str``. OCID of the master key used for encrypting the secret.
- ``compartment_id: str``. OCID of the compartment where the vault is located. This defaults to the compartment of the notebook session when
  used in a Data Science notebook session.

**OracleDBSecretKeeper.save**

``OracleDBSecretKeeper.save`` API serializes and stores the credentials to Vault using the following parameters:

- ``name (str)`` – Name of the secret when saved in the vault.
- ``description (str)`` – Description of the secret when saved in the vault.
- ``freeform_tags (dict, optional)`` – Freeform tags to use when saving the secret in the OCI Console.
- ``defined_tags (dict, optional.)`` – Save the tags under predefined tags in the OCI Console.

The secret content has following information -

- ``user_name``
- ``password``
- ``host``
- ``port``
- ``service_name``
- ``sid``
- ``dsn``

Examples
++++++++

**Saving Database credentials**

.. code:: python3

    import ads
    from ads.secrets.oracledb import OracleDBSecretKeeper

    vault_id = "ocid1.vault.oc1..<unique_ID>"
    key_id = "ocid1.key..<unique_ID>"

    ads.set_auth("resource_principal") # If using resource principal for authentication
    connection_parameters={
         "user_name":"<your user name>",
         "password":"<your password>",
         "service_name":"service_name",
         "host":"<db host>",
         "port":"<db port>",
    }

    oracledb_keeper = OracleDBSecretKeeper(vault_id=vault_id,
                                    key_id=key_id,
                                    **connection_parameters)

    oracledb_keeper.save("oracledb_employee", "My DB credentials", freeform_tags={"schema":"emp"})
    print(oracledb_keeper.secret_id) # Prints the secret_id of the stored credentials

``'ocid1.vaultsecret.oc1..<unique_ID>'``

You can save the vault details in a file for later reference or using it within your code using ``export_vault_details``
API calls. The API currently enables you to export the information as a YAML file or a JSON file.

.. code:: python3

    oracledb_keeper.export_vault_details("my_db_vault_info.json", format="json")

To save as a YAML file:

.. code:: python3

    oracledb_keeper.export_vault_details("my_db_vault_info.yaml", format="yaml")

Loading Credentials
-------------------

**Prerequisite**

- OCID of the secret stored in vault.

**OracleDBSecretKeeper.load_secret**

``OracleDBSecretKeeper.load_secret`` API deserializes and loads the credentials from the vault. You could use this API in one of
the following ways -

Using a ``with`` statement:

.. code:: python3

    with OracleDBSecretKeeper.load_secret('ocid1.vaultsecret.oc1..<unique_ID>') as oracledb_secret:
        print(oracledb_secret['user_name']

Without using a ``with`` statement:

.. code:: python3

    oracledb_secretobj = OracleDBSecretKeeper.load_secret('ocid1.vaultsecret.oc1..<unique_ID>')
    oracledb_secret = oracledb_secretobj.to_dict()
    print(oracledb_secret['user_name'])


``load_secret`` takes following parameters -

- ``source``: Either the file that was exported from ``export_vault_details`` or the OCID of the secret
- ``format``: Optional. If ``source`` is a file, then this value must be ``json`` or ``yaml`` depending on the file format.
- ``export_env``: Default is False. If set to True, the credentials are exported as environment variable when used with
  the ``with`` operator.
- ``export_prefix``: The default name for environment variable is user_name, password, service_name, and wallet_location. You
  can add a prefix to avoid name collision
- ``auth``: Provide overriding authorization information if the authorization information is different from the ``ads.set_auth`` setting.

Examples
++++++++

**Access Credentials with a With Statement**

.. code:: python3

    import ads
    ads.set_auth('resource_principal') # If using resource principal authentication
    from ads.secrets.oracledb import OracleDBSecretKeeper

    with OracleDBSecretKeeper.load_secret(
                "ocid1.vaultsecret.oc1..<unique_ID>"
            ) as oracledb_creds2:
        print (oracledb_creds2["user_name"]) # Prints the user name

    print (oracledb_creds2["user_name"]) # Prints nothing. The credentials are cleared from the dictionary outside the ``with`` block


**Contextually Export Credentials as an Environment Variable Using a With Statement**

To expose credentials as an environment variable, set ``export_env=True``. The following keys are exported:

+-------------------+---------------------------+
| Secret attribute  | Environment Variable Name |
+===================+===========================+
| user_name         | user_name                 |
+-------------------+---------------------------+
| password          | password                  |
+-------------------+---------------------------+
| host              | host                      |
+-------------------+---------------------------+
| port              | port                      |
+-------------------+---------------------------+
| service user_name | service_name              |
+-------------------+---------------------------+
| sid               | sid                       |
+-------------------+---------------------------+
| dsn               | dsn                       |
+-------------------+---------------------------+

.. code:: python3

    import os
    import ads

    ads.set_auth('resource_principal') # If using resource principal authentication
    from ads.secrets.oracledb import OracleDBSecretKeeper

    with OracleDBSecretKeeper.load_secret(
                "ocid1.vaultsecret.oc1..<unique_ID>",
                export_env=True
            ):
        print(os.environ.get("user_name")) # Prints the user name

    print(os.environ.get("user_name")) # Prints nothing. The credentials are cleared from the dictionary outside the ``with`` block

**Avoiding Name Collision with Your Existing Environment Variables**

You can avoid name collision by setting a prefix string using ``export_prefix`` along with ``export_env=True``. For example, if you set prefix as ``myprocess``,
then the keys are exported as:

+-------------------+---------------------------+
| Secret attribute  | Environment Variable Name |
+===================+===========================+
| user_name         | myprocess.user_name       |
+-------------------+---------------------------+
| password          | myprocess.password        |
+-------------------+---------------------------+
| host              | myprocess.host            |
+-------------------+---------------------------+
| port              | myprocess.port            |
+-------------------+---------------------------+
| service user_name | myprocess.service_name    |
+-------------------+---------------------------+
| sid               | myprocess.sid             |
+-------------------+---------------------------+
| dsn               | myprocess.dsn             |
+-------------------+---------------------------+

.. code:: python3

    import os
    import ads

    ads.set_auth('resource_principal') # If using resource principal authentication
    from ads.secrets.oracledb import OracleDBSecretKeeper

    with OracleDBSecretKeeper.load_secret(
                "ocid1.vaultsecret.oc1..<unique_ID>",
                export_env=True,
                export_prefix="myprocess"
            ):
        print(os.environ.get("myprocess.user_name")) # Prints the user name

    print(os.environ.get("myprocess.user_name")) # Prints nothing. The credentials are cleared from the dictionary outside the ``with`` block








