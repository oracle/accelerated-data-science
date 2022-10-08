MySQL
*****

To connect to a MySQL Database, you need the following:

* hostname
* password
* port, the default is 3306
* user name

The ``MySQLDBSecretKeeper`` class saves the MySQL database credentials to the OCI Vault service.

See `API Documentation <../../ads.secrets.html#ads.secrets.mysqldb.MySQLDBSecretKeeper>`__ for more details 


Save Credentials
==================

``MySQLDBSecretKeeper``
-----------------------

The ``MySQLDBSecretKeeper`` constructor has the following parameters:

* ``compartment_id`` (str): OCID of the compartment where the vault is located. Defaults to the compartment of the notebook session when used in a Data Science notebook session.
* ``database`` (str, optional)): The database name if available.
* ``host`` (str): The hostname of the database.
* ``key_id`` (str): OCID of the master key used for encrypting the secret.
* ``password`` (str): The password of the database.
* ``port`` (str, optional). Default 3306): Port number of the database service.
* ``user_name`` (str): The user name to be stored.
* ``vault_id`` (str): OCID of the vault.

Save
^^^^

The ``MySQLDBSecretKeeper.save`` API serializes and stores the credentials to the vault using the following parameters:

* ``defined_tags`` (dict, optional): Save the tags under predefined tags in the OCI Console.
* ``description`` (str): Description of the secret when saved in the vault.
* ``freeform_tags`` (dict, optional): Freeform tags to be used for saving the secret in the OCI Console.
* ``name`` (str): Name of the secret when saved in the vault.

The secret has the following informatio:

* ``database``
* ``host``
* ``password``
* ``port``
* ``user_name``

Examples
--------

Save Credentials
^^^^^^^^^^^^^^^^

.. code-block:: python3

    import ads
    from ads.secrets.mysqldb import MySQLDBSecretKeeper

    vault_id = "ocid1.vault..<unique_ID>"
    key_id = "ocid1.key..<unique_ID>"

    ads.set_auth("resource_principal") # If using resource principal for authentication
    connection_parameters={
         "user_name":"<your user name>",
         "password":"<your password>",
         "service_name":"service_name",
         "host":"<db host>",
         "port":"<db port>",
    }

    mysqldb_keeper = MySQLDBSecretKeeper(vault_id=vault_id,
                                    key_id=key_id,
                                    **connection_parameters)

    mysqldb_keeper.save("mysqldb_employee", "My DB credentials", freeform_tags={"schema":"emp"})
    print(mysqldb_keeper.secret_id) # Prints the secret_id of the stored credentials

``'ocid1.vaultsecret..<unique_ID>'``

You can save the vault details in a file for later reference, or use it in your code using ``export_vault_details`` API calls. The API currently enables you to export the information as a YAML file or a JSON file.

.. code-block:: python3

    mysqldb_keeper.export_vault_details("my_db_vault_info.json", format="json")

Save as a YAML File
^^^^^^^^^^^^^^^^^^^

.. code-block:: python3

    mysqldb_keeper.export_vault_details("my_db_vault_info.yaml", format="yaml")

Load Credentials
================

Load
----

The ``MySQLDBSecretKeeper.load_secret()`` API deserializes and loads the credentials from the vault. You could use this API in one of the following ways:

Using a ``with`` Statement
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python3

    with MySQLDBSecretKeeper.load_secret('ocid1.vaultsecret..<unique_ID>') as mysqldb_secret:
        print(mysqldb_secret['user_name']

Without Using a ``with`` Statement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python3

    mysqldb_secretobj = MySQLDBSecretKeeper.load_secret('ocid1.vaultsecret..<unique_ID>')
    mysqldb_secret = mysqldb_secretobj.to_dict()
    print(mysqldb_secret['user_name'])


The ``.load_secret()`` method has the following parameters:

- ``auth``: Provide overriding ``auth`` information if the ``auth`` information is different from the ``ads.set_auth`` setting.
- ``export_env``: The default is ``False``. If set to ``True``, the credentials are exported as environment variabled when used with the ``with`` operator.
- ``export_prefix``: The default name for environment variable is user_name, password, service_name. and wallet_location. You can add a prefix to avoid name collision.
- ``format``: (Optional) If ``source`` is a file, then this value must be ``json`` or ``yaml`` depending on the file format.
- ``source``: Either the file that was exported from ``export_vault_details``, or the OCID of the secret.

Examples
--------

Using a ``with`` Statement
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python3

    import ads
    ads.set_auth('resource_principal') # If using resource principal authentication
    from ads.secrets.mysqldb import MySQLDBSecretKeeper

    with MySQLDBSecretKeeper.load_secret(
                "ocid1.vaultsecret..<unique_ID>"
            ) as mysqldb_creds2:
        print (mysqldb_creds2["user_name"]) # Prints the user name

    print (mysqldb_creds2["user_name"]) # Prints nothing. The credentials are cleared from the dictionary outside the ``with`` block


Export the Environment Variables Using a ``with`` Statement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To expose credentials as an environment variable, set ``export_env=True``. The following keys are exported:

+------------------+---------------------------+
| Secret attribute | Environment Variable Name |
+==================+===========================+
| user_name        | user_name                 |
+------------------+---------------------------+
| password         | password                  |
+------------------+---------------------------+
| host             | host                      |
+------------------+---------------------------+
| port             | port                      |
+------------------+---------------------------+
| database         | database                  |
+------------------+---------------------------+

.. code-block:: python3

    import os
    import ads

    ads.set_auth('resource_principal') # If using resource principal authentication
    from ads.secrets.mysqldb import MySQLDBSecretKeeper

    with MySQLDBSecretKeeper.load_secret(
                "ocid1.vaultsecret..<unique_ID>",
                export_env=True
            ):
        print(os.environ.get("user_name")) # Prints the user name

    print(os.environ.get("user_name")) # Prints nothing. The credentials are cleared from the dictionary outside the ``with`` block

You can avoid name collisions by setting a prefix string using ``export_prefix`` along with ``export_env=True``. For example, if you set prefix as ``myprocess``, then the exported keys are:

+------------------+---------------------------+
| Secret attribute | Environment Variable Name |
+==================+===========================+
| user_name        | myprocess.user_name       |
+------------------+---------------------------+
| password         | myprocess.password        |
+------------------+---------------------------+
| host             | myprocess.host            |
+------------------+---------------------------+
| port             | myprocess.port            |
+------------------+---------------------------+
| database         | myprocess.database        |
+------------------+---------------------------+

.. code-block:: python3

    import os
    import ads

    ads.set_auth('resource_principal') # If using resource principal authentication
    from ads.secrets.mysqldb import MySQLDBSecretKeeper

    with MySQLDBSecretKeeper.load_secret(
                "ocid1.vaultsecret..<unique_ID>",
                export_env=True,
                export_prefix="myprocess"
            ):
        print(os.environ.get("myprocess.user_name")) # Prints the user name

    print(os.environ.get("myprocess.user_name")) # Prints nothing. The credentials are cleared from the dictionary outside the ``with`` block

