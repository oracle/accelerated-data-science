Auth Token
**********

The ``AuthTokenSecretKeeper`` helps you to save the Auth Token or Access Token string to the OCI Vault service.

See `API Documentation <../../ads.secrets.html#ads.secrets.auth_token.AuthTokenSecretKeeper>`__ for more details 

Save Credentials
================

``AuthTokenSecretKeeper``
-------------------------

The ``AuthTokenSecretKeeper`` constructor takes the following parameters:

* ``auth_token`` (str): Provide the Auth Token or Access Token string to be stored
* ``vault_id`` (str): ocid of the vault
* ``key_id`` (str): ocid of the master key used for encrypting the secret
* ``compartment_id`` (str, optional): Default is None. ocid of the compartment where the vault is located. This will be defaulted to the compartment of the Notebook session, if used within a OCI Data Science notebook session.

Save
^^^^

The ``AuthTokenSecretKeeper.save`` API serializes and stores the credentials to Vault. It takes following parameters -

* ``name`` (str): Name of the secret when saved in the vault.
* ``description`` (str): Description of the secret when saved in the vault.
* ``freeform_tags`` (dict, optional): Freeform tags to use when saving the secret in the OCI Console.
* ``defined_tags`` (dict, optional.): Save the tags under predefined tags in the OCI Console.

The secret has following information: 

* auth_token

Examples
--------

Save Auth Token
^^^^^^^^^^^^^^^

.. code-block:: python3

    import ads
    from ads.secrets.auth_token import AuthTokenSecretKeeper

    ads.set_auth('resource_principal') # If using resource principal authentication

    ocid_vault = "ocid1.vault...<unique_ID>"
    ocid_master_key = "ocid1.key..<unique_ID>"
    ocid_mycompartment = "ocid1.compartment..<unique_ID>"

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

You can save the vault details in a file for later reference or using it within your code using ``export_vault_details`` API. The API currently let us export the information as a ``yaml`` file or a ``json`` file.

.. code-block:: python3

    authtoken2.export_vault_details("my_db_vault_info.json", format="json")

Save as a ``yaml`` File
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python3

    authtoken2.export_vault_details("my_db_vault_info.yaml", format="yaml")

Load Credentials
================

Load
----

The ``AuthTokenSecretKeeper.load_secret`` API deserializes and loads the credentials from Vault. You could use this API in one of the following ways:

Using a ``with`` Statement
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python3

    with AuthTokenSecretKeeper.load_secret('ocid1.vaultsecret..<unique_ID>') as authtoken:
        print(authtoken['user_name']

This approach is preferred as the secrets are only available within the code block and it reduces the risk that the variable will be leaked.

Without using a ``with`` Statement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python3

    authtoken = AuthTokenSecretKeeper.load_secret('ocid1.vaultsecret..<unique_ID>')
    authtokendict = authtoken.to_dict()
    print(authtokendict['user_name'])


The ``.load_secret()`` takes the following parameters:

* ``auth``: Provide overriding authorization information if the authorization information is different from the ``ads.set_auth`` setting.
* ``export_env``: Default is False. If set to True, the credentials are exported as environment variable when used with
* ``export_prefix``: The default name for environment variable is user_name, password, service_name, and wallet_location. You can add a prefix to avoid name collision
* ``format``: Optional. If ``source`` is a file, then this value must be ``json`` or ``yaml`` depending on the file format.
* ``source``: Either the file that was exported from ``export_vault_details`` or the OCID of the secret
* the ``with`` operator.

Examples
--------

Using a ``with`` Statement
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python3

    import ads
    from ads.secrets.auth_token import AuthTokenSecretKeeper

    ads.set_auth('resource_principal') # If using resource principal authentication

    with AuthTokenSecretKeeper.load_secret(source="ocid1.vaultsecret..<unique_ID",
                                   ) as authtoken:
        import os
        print(f"Credentials inside `authtoken` object:  {authtoken}")

``Credentials inside `authtoken` object:  {'auth_token': '<your_auth_token>'}``


Export to Environment Variables Using a ``with`` Statement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To expose credentials through environment variable, set ``export_env=True``. The following keys are exported -

+------------------+---------------------------+
| Secret attribute | Environment Variable Name |
+==================+===========================+
| auth_token       | auth_token                |
+------------------+---------------------------+

.. code-block:: python3

    import ads
    from ads.secrets.auth_token import AuthTokenSecretKeeper
    import os

    ads.set_auth('resource_principal') # If using resource principal authentication

    with AuthTokenSecretKeeper.load_secret(
                source="ocid1.vaultsecret..<unique_ID>",
                export_env=True
            ):
        print(os.environ.get("auth_token")) # Prints the auth token

    print(os.environ.get("auth_token")) # Prints nothing. The credentials are cleared from the dictionary outside the ``with`` block

You can avoid name collisions by setting the prefix string using ``export_prefix`` along with ``export_env=True``. For example, if you set the prefix to ``kafka``, the exported keys are:

+------------------+---------------------------+
| Secret attribute | Environment Variable Name |
+==================+===========================+
| auth_token       | kafka.auth_token          |
+------------------+---------------------------+


.. code-block:: python3

    import ads
    from ads.secrets.auth_token import AuthTokenSecretKeeper
    import os

    ads.set_auth('resource_principal') # If using resource principal authentication

    with AuthTokenSecretKeeper.load_secret(
                source="ocid1.vaultsecret..<unique_ID>",
                export_env=True,
                export_prefix="kafka"
            ):
        print(os.environ.get("kafka.auth_token")) # Prints the auth token

    print(os.environ.get("kafka.auth_token")) # Prints nothing. The credentials are cleared from the dictionary outside the ``with`` block


