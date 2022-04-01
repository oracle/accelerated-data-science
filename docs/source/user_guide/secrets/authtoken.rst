Auth Token
==========

``AuthTokenSecretKeeper`` helps you to save the Auth Token or Access Token string to the OCI Vault service.

Saving Credentials
------------------

**Prerequisite**

- OCID of the Vault created on OCI console
- OCID of the master key that will be used for encrypting the secret content stored inside Vault
- OCID of the compartment where the Vault resides. This will be defaulted to the compartment of the Notebook session, if
  used within a OCI Data Science notebook session.

**AuthTokenSecretKeeper**

AuthTokenSecretKeeper takes following constructor parameter -

- ``auth_token: str``. Provide the Auth Token or Access Token string to be stored
- ``vault_id: str``. ocid of the vault
- ``key_id: str``. ocid of the master key used for encrypting the secret
- ``compartment_id: (str, optional)``. Default is None. ocid of the compartment where the vault is located. This will be defaulted to the compartment of the Notebook session, if
  used within a OCI Data Science notebook session.


**AuthTokenSecretKeeper.save**

``AuthTokenSecretKeeper.save`` API serializes and stores the credentials to Vault. It takes following parameters -

- ``name (str)`` – Name of the secret when saved in the vault.
- ``description (str)`` – Description of the secret when saved in the vault.
- ``freeform_tags (dict, optional)`` – Freeform tags to use when saving the secret in the OCI Console.
- ``defined_tags (dict, optional.)`` – Save the tags under predefined tags in the OCI Console.

The secret content has following information -

- auth_token

Examples
++++++++

**Saving Auth Token string**

.. code:: python3

    import ads
    from ads.secrets.auth_token import AuthTokenSecretKeeper

    ads.set_auth('resource_principal') # If using resource principal authentication

    ocid_vault = "ocid1.vault.oc1...<unique_ID>"
    ocid_master_key = "ocid1.key.oc1..<unique_ID>"
    ocid_mycompartment = "ocid1.compartment.oc1..<unique_ID>"

    authtoken2 = AuthTokenSecretKeeper(
                    vault_id=ocid_vault,
                    key_id=ocid_master_key,
                    compartment_id=ocid_mycompartment,
                    auth_token="<your_auth_token>"
                   ).save(
                        "my_xyz_auth_token2",
                        "This is my key for git repo xyz",
                        freeform_tags={"gitrepo":"xyz"}
                    )
    print(authtoken2.secret_id)

You can save the vault details in a file for later reference or using it within your code using ``export_vault_details``
API. The API currently let us export the information as a ``yaml`` file or a ``json`` file.

.. code:: python3

    authtoken2.export_vault_details("my_db_vault_info.json", format="json")

To save as a ``yaml`` file

.. code:: python3

    authtoken2.export_vault_details("my_db_vault_info.yaml", format="yaml")

Loading Credentials
-------------------

**Prerequisite**

- OCID of the secret stored in OCI Vault.

**AuthTokenSecretKeeper.load_secret**

``AuthTokenSecretKeeper.load_secret`` API deserializes and loads the credentials from Vault. You could use this API in one of
the following ways -

Option 1: Using ``with`` statement

.. code:: python3

    with AuthTokenSecretKeeper.load_secret('ocid1.vaultsecret.oc1..<unique_ID>') as authtoken:
        print(authtoken['user_name']

Option 2: Without using ``with`` statement.

.. code:: python3

    authtoken = AuthTokenSecretKeeper.load_secret('ocid1.vaultsecret.oc1..<unique_ID>')
    authtokendict = authtoken.to_dict()
    print(authtokendict['user_name'])


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

**Access credentials within With Statement**

.. code:: python3

    import ads
    from ads.secrets.auth_token import AuthTokenSecretKeeper

    ads.set_auth('resource_principal') # If using resource principal authentication

    with AuthTokenSecretKeeper.load_secret(source="ocid1.vaultsecret.oc1...<unique_ID",
                                   ) as authtoken:
        import os
        print(f"Credentials inside `authtoken` object:  {authtoken}")

``Credentials inside `authtoken` object:  {'auth_token': '<your_auth_token>'}``


**Contextually export credentials as environment variable using With statement**

To expose credentials through environment variable, set ``export_env=True``. The following keys are exported -

+------------------+---------------------------+
| Secret attribute | Environment Variable Name |
+==================+===========================+
| auth_token       | auth_token                |
+------------------+---------------------------+

.. code:: python3

    import ads
    from ads.secrets.auth_token import AuthTokenSecretKeeper
    import os

    ads.set_auth('resource_principal') # If using resource principal authentication

    with AuthTokenSecretKeeper.load_secret(
                source="ocid1.vaultsecret.oc1...<unique_ID>",
                export_env=True
            ):
        print(os.environ.get("auth_token")) # Prints the auth token

    print(os.environ.get("auth_token")) # Prints nothing. The credentials are cleared from the dictionary outside the ``with`` block

**Avoding name collision with your existing environment variables**

Name collision can be avoided by providing a prefix string through ``export_prefix`` along with ``export_env=True``. Example, if you set prefix as ``kafka``
The keys are exported as -

+------------------+---------------------------+
| Secret attribute | Environment Variable Name |
+==================+===========================+
| auth_token       | kafka.auth_token          |
+------------------+---------------------------+


.. code:: python3

    import ads
    from ads.secrets.auth_token import AuthTokenSecretKeeper
    import os

    ads.set_auth('resource_principal') # If using resource principal authentication

    with AuthTokenSecretKeeper.load_secret(
                source="ocid1.vaultsecret.oc1...<unique_ID>",
                export_env=True,
                export_prefix="kafka"
            ):
        print(os.environ.get("kafka.auth_token")) # Prints the auth token

    print(os.environ.get("kafka.auth_token")) # Prints nothing. The credentials are cleared from the dictionary outside the ``with`` block









