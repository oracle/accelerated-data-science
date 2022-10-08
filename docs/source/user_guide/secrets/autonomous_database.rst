Autonomous Database
*******************

To connect to Autonomous Database you need the following:

* user name
* password
* service name
* `wallet file
  <https://docs.oracle.com/en/cloud/paas/autonomous-database/adbsa/connect-download-wallet.html#GUID-DED75E69-C303-409D-9128-5E10ADD47A35>`_

The ``ADBSecretKeeper`` class saves the ADB credentials to the OCI Vault service.

See `API Documentation <../../ads.secrets.html#ads.secrets.adb.ADBSecretKeeper>`__ for more details 

Save Credentials
================

``ADBSecretKeeper``
-------------------

The ``ADBSecretKeeper`` constructor has the following parameters:

* ``compartment_id`` (str): OCID of the compartment where the vault is located. This defaults to the compartment of the notebook session when used in a Data Science notebook session.
* ``key_id`` (str): OCID of the master key used for encrypting the secret.
* ``password`` (str): The password of the database.
* ``service_name`` (str): Set the service name of the database.
* ``user_name`` (str): The user name to be stored.
* ``vault_id`` (str): OCID of the vault.
* ``wallet_location`` (str): Path to the wallet ZIP file.


Save
^^^^

The ``ADBSecretKeeper.save`` API serializes and stores the credentials to Vault using the following parameters:

- ``defined_tags`` (dict, optional): Default None. Save the tags under predefined tags in the OCI Console.
- ``description`` (str): Description of the secret when saved in Vault.
- ``freeform_tags`` (dict, optional): Default None. Free form tags to use for saving the secret in the OCI Console.
- ``name`` (str): Name of the secret when saved in Vault.
- ``save_wallet`` (bool, optional): Default False. If set to True, then the wallet file is serialized.

When stored without the wallet information, the secret content has following information:

* ``password``
* ``service_name``
* ``user_name``

To store wallet file content, set ``save_wallet`` to ``True``. The wallet content is stored by extracting all the files from the wallet ZIP file, and then each file is stored in the vault as a secret. The list of OCIDs corresponding to each file along with username, password, and service name is stored in a separate secret.  The secret corresponding to each file content has following information:

* filename
* content of the file

A **meta secret** is created to save the username, password, service name, and the secret ids of the files within the wallet file. It has following attributes:

* ``user_name``
* ``password``
* ``wallet_file_name``
* ``wallet_secret_ids``

The wallet file is reconstructed when ``ADBSecretKeeper.load_secret`` is called using the OCID of the **meta secret**.

Examples
--------

Without the Wallet File
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python3

    import ads
    ads.set_auth('resource_principal') # If using resource principal authentication
    from ads.secrets.adb import ADBSecretKeeper

    connection_parameters={
        "user_name":"admin",
        "password":"<your_password>",
        "service_name":"service_high",
        "wallet_location":"/home/datascience/Wallet_--------.zip"
    }

    ocid_vault = "ocid1.vault..<unique_ID>"
    ocid_master_key = "ocid1.key..<unique_ID>"
    ocid_mycompartment = "ocid1.compartment..<unique_ID>"

    adw_keeper = ADBSecretKeeper(vault_id=ocid_vault,
                                key_id=ocid_master_key,
                                compartment_id=ocid_mycompartment,
                                **connection_parameters)

    # Store the credentials without storing the wallet file
    adw_keeper.save("adw_employee_att2", "My DB credentials", freeform_tags={"schema":"emp"})
    print(adw_keeper.secret_id)

``'ocid1.vaultsecret..<unique_ID>'``

With the Wallet File
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python3

    import ads
    ads.set_auth('resource_principal') # If using resource principal authentication
    from ads.secrets.adb import ADBSecretKeeper

    connection_parameters={
        "user_name":"admin",
        "password":"<your_password>",
        "service_name":"service_high",
        "wallet_location":"/home/datascience/Wallet_--------.zip"
    }

    ocid_vault = "ocid1.vault..<unique_ID>"
    ocid_master_key = "ocid1.key..<unique_ID>"
    ocid_mycompartment = "ocid1.compartment..<unique_ID>"

    adw_keeper = ADBSecretKeeper(vault_id=ocid_vault,
                                key_id=ocid_master_key,
                                compartment_id=ocid_mycompartment,
                                **connection_parameters)

    # Set `save_wallet`=True to save wallet file

    adw_keeper.save("adw_employee_att2",
        "My DB credentials",
        freeform_tags={"schema":"emp"},
        save_wallet=True
    )

    print(adw_keeper.secret_id)

``'ocid1.vaultsecret..<unique_ID>'``

You can save the vault details in a file for later reference or using it within your code using ``export_vault_details`` API calls. The API currently enables you to export the information as a YAML file or a JSON file.

.. code-block:: python3

    adw_keeper.export_vault_details("my_db_vault_info.json", format="json")

To save as a YAML file:

.. code-block:: python3

    adw_keeper.export_vault_details("my_db_vault_info.yaml", format="yaml")

Load Credentials
================

Load
----

The ``ADBSecretKeeper.load_secret`` API deserializes and loads the credentials from Vault. You could use this API in one of
the following ways: 

Using a ``with`` Statement
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python3

    with ADBSecretKeeper.load_secret('ocid1.vaultsecret..<unique_ID>') as adwsecret:
        print(adwsecret['user_name'])

This approach is preferred as the secrets are only available within the code block and it reduces the risk that the variable will be leaked.

Without using a ``with`` Statement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python3

    adwsecretobj = ADBSecretKeeper.load_secret('ocid1.vaultsecret..<unique_ID>')
    adwsecret = adwsecretobj.to_dict()
    print(adwsecret['user_name'])


The ``.load_secret()`` method has the following parameters:

* ``auth``: Provide overriding authorization information if the authorization information is different from the ``ads.set_auth`` setting.
* ``export_env``: Default is False. If set to True, the credentials are exported as environment variable when used with the ``with`` operator.
* ``export_prefix``: The default name for environment variable is user_name, password, service_name, and wallet_location. You can add a prefix to avoid name collision
* ``format``: Optional. If ``source`` is a file, then this value must be ``json`` or ``yaml`` depending on the file format.
* ``source``: Either the file that was exported from ``export_vault_details`` or the OCID of the secret
* ``wallet_dir``: Optional. Directory path where the wallet zip file will be saved after the contents are retrieved from Vault. If wallet content is not available in the provided secret OCID, this attribute is ignored.
* ``wallet_location``: Optional. Path to the local wallet zip file. If vault secret does not have wallet file content, set this variable so that it will be available in the exported credential. If provided, this path takes precedence over the wallet file information in the secret.

If the wallet file was saved in the vault, then the ZIP file of the same name is created by the ``.load_secret()`` method. By default the ZIP file is created in the working directory. To update the location, you can set the directory path with ``wallet_dir``.

Examples
--------

Using a ``with`` Statement
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python3

    import ads
    ads.set_auth('resource_principal') # If using resource principal authentication
    from ads.secrets.adb import ADBSecretKeeper

    with ADBSecretKeeper.load_secret(
                "ocid1.vaultsecret..<unique_ID>"
            ) as adw_creds2:
        print (adw_creds2["user_name"]) # Prints the user name

    print (adw_creds2["user_name"]) # Prints nothing. The credentials are cleared from the dictionary outside the ``with`` block

Export to Environment Variables Using a ``with`` Statement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To expose credentials as an environment variable, set ``export_env=True``. The following keys are exported:

+------------------+---------------------------+
| Secret attribute | Environment Variable Name |
+==================+===========================+
| user_name        | user_name                 |
+------------------+---------------------------+
| password         | password                  |
+------------------+---------------------------+
| service_name     | service_name              |
+------------------+---------------------------+
| wallet_location  | wallet_location           |
+------------------+---------------------------+

.. code-block:: python3

    import os
    import ads

    ads.set_auth('resource_principal') # If using resource principal authentication
    from ads.secrets.adb import ADBSecretKeeper

    with ADBSecretKeeper.load_secret(
                "ocid1.vaultsecret..<unique_ID>",
                export_env=True
            ):
        print(os.environ.get("user_name")) # Prints the user name

    print(os.environ.get("user_name")) # Prints nothing. The credentials are cleared from the dictionary outside the ``with`` block

You can avoid name collisions by setting a prefix string using ``export_prefix`` along with ``export_env=True``. For example, if you set the prefix to ``myprocess``, then the exported keys are:

+------------------+---------------------------+
| Secret attribute | Environment Variable Name |
+==================+===========================+
| user_name        | myprocess.user_name       |
+------------------+---------------------------+
| password         | myprocess.password        |
+------------------+---------------------------+
| service_name     | myprocess.service_name    |
+------------------+---------------------------+
| wallet_location  | myprocess.wallet_location |
+------------------+---------------------------+

.. code-block:: python3

    import os
    import ads

    ads.set_auth('resource_principal') # If using resource principal authentication
    from ads.secrets.adb import ADBSecretKeeper

    with ADBSecretKeeper.load_secret(
                "ocid1.vaultsecret..<unique_ID>",
                export_env=True,
                export_prefix="myprocess"
            ):
        print(os.environ.get("myprocess.user_name")) # Prints the user name

    print(os.environ.get("myprocess.user_name")) # Prints nothing. The credentials are cleared from the dictionary outside the ``with`` block

Wallet File Location
^^^^^^^^^^^^^^^^^^^^

You can set wallet file location when wallet file is not part of the stored vault secret. To specify a local wallet ZIP file, set the path to the ZIP file with ``wallet_location``:

.. code-block:: python3

    import ads
    ads.set_auth('resource_principal') # If using resource principal authentication
    from ads.secrets.adb import ADBSecretKeeper

    with ADBSecretKeeper.load_secret(
                "ocid1.vaultsecret..<unique_ID>",
                wallet_location="path/to/my/local/wallet.zip"
            ) as adw_creds2:
        print (adw_creds2["wallet_location"]) # Prints `path/to/my/local/wallet.zip`

    print (adw_creds2["wallet_location"]) # Prints nothing. The credentials are cleared from the dictionary outside the ``with`` block

