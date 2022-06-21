.. _configuration-vault:

Vault
*****

The Oracle Cloud Infrastructure Vault is a service that provides management of encryption keys and secret credentials. A vault is a storage container that holds keys and secrets. The Vault service not only secures your secrets it provides a central repository that allows them to be used in different notebooks and shared with only those that need access. No longer will your secrets be stored in code that can accidentally be checked into git repositories.

This notebook demonstrates how to create a vault, a key, and store a secret that is encrypted with that key. It also demonstrates how to retrieve the secret so that it can be used in a notebook. The notebook explains how to update that secret and basic operations, such as listing deleting vaults, keys, and secrets.

This notebook performs CRUD (create, read, update, delete) operations on vaults, keys, and secrets. These are all part of the Vault Service. The account that is using this notebook requires permissions to these resources. The account administrator needs to grant privileges to perform these actions. How the permissions are configured can depend on your tenancy configuration, see the `Vault Service’s permissions documentation <https://docs.cloud.oracle.com/en-us/iaas/Content/Identity/Reference/keypolicyreference.htm>`__ for details. The `Vault Service’s common policies <https://docs.cloud.oracle.com/en-us/iaas/Content/Identity/Concepts/commonpolicies.htm#sec-admins-manage-vaults-keys>`__ are:

::

   allow group <group> to manage vaults in compartment <compartment>
   allow group <group> to manage keys in compartment <compartment>
   allow group <group> to manage secret-family in compartment <compartment>

**Introduction to the Vault Service**

The `Oracle Cloud Infrastructure Vault <https://docs.cloud.oracle.com/en-us/iaas/Content/KeyManagement/Concepts/keyoverview.htm>`__ lets you centrally manage the encryption keys that protect your data and the secret credentials that you use to securely access resources.

Vaults securely store master encryption keys and secrets that you might otherwise store in configuration files or in code.

Use the Vault service to exercise control over the lifecycle keys and secrets. Integration with Oracle Cloud Infrastructure Identity and Access Management (IAM) lets you control who and what services can access which keys and secrets and what they can do with those resources.  The Oracle Cloud Infrastructure Audit integration gives you a way to monitor key and secret use. Audit tracks administrative actions on vaults, keys, and secrets.

Keys are stored on highly available and durable hardware security modules (HSM) that meet Federal Information Processing Standards (FIPS) The main use case for a data scientist is to store a secret, such as an SSH key, database password, or some other credential. To do this, a vault and key are required. By default, this notebook creates these resources. However, the ``vault_id`` and ``key_id`` variables can be updated with vault and key OCIDs to use existing resources.

.. code-block:: python3

    # Select the configuration file to connect to Oracle Cloud Infrastructure resources
    config = from_file(path.join(path.expanduser("~"), ".oci", "config"), "DEFAULT")

    # Select the compartment to create the secrets in.
    # Use the notebook compartment by default
    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']

    # Enter a vault OCID. Otherwise, one is created.
    vault_id = "<vault_id>"
    # Enter a KMS OCID to encrypt the secret. Otherwise, one is created
    key_id = "<key_id>"


For the purposes of this notebook, a secret is stored. The secret is the credentials needed to access a database. The notebook is designed so that any secret can be stored as long as it is in the form of a dictionary. To store your secret, just modify the dictionary.

.. code-block:: python3

    # Sample credentials that are going to be stored.
    credential = {'database_name': 'databaseName_high',
                  'username': 'admin',
                  'password': 'MySecretPassword',
                  'database_type': 'oracle'}

Note, to connect to an Oracle database the `database_name` value should be its connection identifier. You can find the connection identifier by extracting the credential wallet zip file and opening the `tnsnames.ora` file (connection_identifier = (...)). Usually the connection identifier will end with `_high`, `_medium` or `_low` i.e. `'MyDatabaseName_high'`.

**Create a Vault**

To store a secret, a key is needed to encrypt and decrypt the secret.  This key and secret are stored in a vault. The code in the following cell creates a vault if you have not specified an OCID in the ``vault_id`` variable. The ``KmsVaultClient`` class takes a configuration object and establishes a connection to the key management service (KMS). Communication with ``KmsVaultClient`` is asynchronous.  For the purpose of this notebook, it is better to have synchronous communication so the ``KmsVaultClient`` are wrapped in a ``KmsVaultClientCompositeOperations`` object.

The details of the vault are specified using an object of the ``CreateVaultDetails`` type. A compartment ID must be provided along with the properties of the vault. For the purposes of this notebook, the
vault’s display name is ``DataScienceVault_`` and a random string because the names of a vault must be unique. This value can be changed to fit your individual needs.

.. code-block:: python3

    if vault_id == "<vault_id>":
        # Create a VaultClientCompositeOperations for composite operations.
        vault_client = KmsVaultClientCompositeOperations(KmsVaultClient(config))

        # Create vault_details object for use in creating the vault.
        vault_details = CreateVaultDetails(compartment_id=compartment_id,
            vault_type=oci.key_management.models.Vault.VAULT_TYPE_DEFAULT,
            display_name="DataScienceVault_{}".format(str(uuid.uuid4())[-6:]))

        # Vault creation is asynchronous; Create the vault and wait until it becomes active.
        print("Creating vault...", end='')
        vault = vault_client.create_vault_and_wait_for_state(vault_details,
                    wait_for_states=[oci.vault.models.Secret.LIFECYCLE_STATE_ACTIVE]).data
        vault_id = vault.id
        print('Done')
        print("Created vault: {}".format(vault_id))
    else:
        # Get the vault using the vault OCID.
        vault = KmsVaultClient(config).get_vault(vault_id=vault_id).data
        print("Using vault: {}".format(vault.id))


.. parsed-literal::

    Creating vault...Done
    Created vault: ocid1.vault..<unique_ID>


**Create a Key**

The secret is encrypted and decrypted using an AES key. The code in the following cell creates a key if you have not specified an OCID in the
``key_id`` variable. The ``KmsManagementClient`` class takes a configuration object and the endpoint for the vault that is going to be
used to store the key. It establishes a connection to the KMS. Communication with ``KmsManagementClient`` is asynchronous. For the
purpose of this notebook, it is better to have synchronous communication so the ``KmsManagementClient`` is wrapped in a
``KmsManagementClientCompositeOperations`` object.

The details of the key are specified using an object of type ``CreateKeyDetails``. A compartment OCID must be provided along with the
properties of the key. The ``KeyShape`` class defines the properties of the key. In this example, it is a 32-bit AES key.

For the purposes of this notebook, the key’s display name is ``DataScienceKey_`` and a random string because the names of a key must
be unique. This value can be changed to fit your individual needs.

.. code-block:: python3

    if key_id == "<key_id>":
        # Create a vault management client using the endpoint in the vault object.
        vault_management_client = KmsManagementClientCompositeOperations(
            KmsManagementClient(config, service_endpoint=vault.management_endpoint))

        # Create key_details object that needs to be passed when creating key.
        key_details = CreateKeyDetails(compartment_id=compartment_id,
            display_name="DataScienceKey_{}".format(str(uuid.uuid4())[-6:]),
            key_shape=KeyShape(algorithm="AES", length=32))

        # Vault creation is asynchronous; Create the vault and wait until it becomes active.
        print("Creating key...", end='')
        key = vault_management_client.create_key_and_wait_for_state(key_details,
                  wait_for_states=[oci.key_management.models.Key.LIFECYCLE_STATE_ENABLED]).data
        key_id = key.id
        print('Done')
        print("Created key: {}".format(key_id))
    else:
        print("Using key: {}".format(key_id))


.. parsed-literal::

    Creating key...Done
    Created key: ocid1.key..<unique_ID>


**Secret**

**Store a Secret**

The code in the following cell creates a secret that is to be stored. The variable ``credential`` is a dictionary and contains the information
that is to be stored. The UDF ``dict_to_secret`` takes a Python dictionary, converts it to a JSON string, and then Base64 encodes it.
This string is what is to be stored as a secret so the secret can be parsed by any system that may need it.

The ``VaultsClient`` class takes a configuration object and establishes a connection to the Vault service. Communication with ``VaultsClient``
is asynchronous. For the purpose of this notebook, it is better to have synchronous communication so ``VaultsClient`` is wrapped in a
``VaultsClientCompositeOperations`` object.

The contents of the secret are stored in a ``Base64SecretContentDetails`` object. This object contains information
about the encoding being used, the stage to be used,and most importantly the payload (the secret). The ``CreateSecretDetails`` class is used to
wrap the ``Base64SecretContentDetails`` object and also specify other properties about the secret. It requires the compartment OCID, the vault
that is to store the secret, and the key to use to encrypt the secret. For the purposes of this notebook, the secret’s display name is
``DataScienceSecret_`` and a random string because the names of a secret must be unique. This value can be changed to fit your individual needs.

.. code-block:: python3

    # Encode the secret.
    secret_content_details = Base64SecretContentDetails(
        content_type=oci.vault.models.SecretContentDetails.CONTENT_TYPE_BASE64,
        stage=oci.vault.models.SecretContentDetails.STAGE_CURRENT,
        content=dict_to_secret(credential))

    # Bundle the secret and metadata about it.
    secrets_details = CreateSecretDetails(
            compartment_id=compartment_id,
            description = "Data Science service test secret",
            secret_content=secret_content_details,
            secret_name="DataScienceSecret_{}".format(str(uuid.uuid4())[-6:]),
            vault_id=vault_id,
            key_id=key_id)

    # Store secret and wait for the secret to become active.
    print("Creating secret...", end='')
    vaults_client_composite = VaultsClientCompositeOperations(VaultsClient(config))
    secret = vaults_client_composite.create_secret_and_wait_for_state(
                 create_secret_details=secrets_details,
                 wait_for_states=[oci.vault.models.Secret.LIFECYCLE_STATE_ACTIVE]).data
    secret_id = secret.id
    print('Done')
    print("Created secret: {}".format(secret_id))


.. parsed-literal::

    Creating secret...Done
    Created secret: ocid1.vaultsecret..<unique_ID>


**Retrieve a Secret**

The ``SecretsClient`` class takes a configuration object. The ``get_secret_budle`` method takes the secret’s OCID and returns a
``Response`` object. Its ``data`` attribute returns ``SecretBundle`` object. This has an attribute ``secret_bundle_content`` that has the
object ``Base64SecretBundleContentDetails`` and the ``content`` attribute of this object has the actual secret. This returns the Base64
encoded JSON string that was created with the ``dict_to_secret`` function. The process can be reversed with the ``secret_to_dict``
function. This will return a dictionary with the secrets. 

.. code-block:: python3

    secret_bundle = SecretsClient(config).get_secret_bundle(secret_id)
    secret_content = secret_to_dict(secret_bundle.data.secret_bundle_content.content)

    print(secret_content)


.. parsed-literal::

    {'database': 'datamart', 'username': 'admin', 'password': 'MySecretPassword'}


**Update a Secret**

Secrets are immutable but it is possible to update them by creating new versions. In the code in the following cell, the ``credential`` object
updates the ``password`` key. To update the secret, a ``Base64SecretContentDetails`` object must be created. The process is
the same as previously described in the `Store a Secret <#store_secret>`__ section. However, instead of using a
``CreateSecretDetails`` object, an ``UpdateSecretDetails`` object is used and only the information that is being changed is passed in.

Note that the OCID of the secret does not change. A new secret version is created and the old secret is rotated out of use, but it may still be
available depending on the tenancy configuration.

The code in the following cell updates the secret. It then prints the OCID of the old secret and the new secret (they will be the same). It
also retrieves the updated secret, converts it into a dictionary, and prints it. This shows that the password was actually updated.

.. code-block:: python3

    # Update the password in the secret.
    credential['password'] = 'UpdatedPassword'

    # Encode the secret.
    secret_content_details = Base64SecretContentDetails(
        content_type=oci.vault.models.SecretContentDetails.CONTENT_TYPE_BASE64,
        stage=oci.vault.models.SecretContentDetails.STAGE_CURRENT,
        content=dict_to_secret(credential))

    # Store the details to update.
    secrets_details = UpdateSecretDetails(secret_content=secret_content_details)

    #Create new secret version and wait for the new version to become active.
    secret_update = vaults_client_composite.update_secret_and_wait_for_state(
        secret_id,
        secrets_details,
        wait_for_states=[oci.vault.models.Secret.LIFECYCLE_STATE_ACTIVE]).data

    # The secret OCID does not change.
    print("Orginal Secret OCID: {}".format(secret_id))
    print("Updated Secret OCID: {}".format(secret_update.id))

    ### Read a secret's value.
    secret_bundle = SecretsClient(config).get_secret_bundle(secret_update.id)
    secret_content = secret_to_dict(secret_bundle.data.secret_bundle_content.content)

    print(secret_content)


.. parsed-literal::

    Orginal Secret OCID: ocid1.vaultsecret..<unique_ID>
    Updated Secret OCID: ocid1.vaultsecret..<unique_ID>
    {'database': 'datamart', 'username': 'admin', 'password': 'UpdatedPassword'}


**List Resources**

This section demonstrates how to obtain a list of resources from the vault, key, and secrets

**List Secrets**

The ``list_secrets`` method of the ``VaultsClient`` provides access to all secrets in a compartment. It provides access to all secrets that are in all vaults in a compartment. It returns a ``Response`` object and the ``data`` attribute in that object is a list of ``SecretSummary`` objects.

The ``SecretSummary`` class has the following attributes: 

* ``compartment_id``: Compartment OCID. 
* ``defined_tags``: Oracle defined tags.
* ``description``: Secret description. 
* ``freeform_tags``: User-defined tags.
* ``id``: OCID of the secret. 
* ``key_id``: OCID of the key used to encrypt and decrypt the secret. 
* ``lifecycle_details``: Details about the lifecycle. 
* ``lifecycle_state``: The current lifecycle state, such as ACTIVE and PENDING_DELETION. 
* ``secret_name``: Name of the secret. 
* ``time_created``: Timestamp of when the secret was created. 
* ``time_of_current_version_expiry``: Timestamp of when the secret expires if it is set to expire. 
* ``time_of_deletion``: Timestamp of when the secret is deleted if it is pending deletion. 
* ``vault_id``: Vault OCID that the secret is in.

Note that the ``SecretSummary`` object does not contain the actual secret. It does provide the secret’s OCID that can be used to obtain the secret bundle, which has the secret. See the `retrieving a secret <#retrieve_secret>`__, section.

The following code uses attributes about a secret to display basic information about all the secrets.

.. code-block:: python3

    secrets = VaultsClient(config).list_secrets(compartment_id)
    for secret in secrets.data:
        print("Name: {}\nLifecycle State: {}\nOCID: {}\n---".format(
            secret.secret_name, secret.lifecycle_state,secret.id))


.. parsed-literal::

    Name: DataScienceSecret_fd63db
    Lifecycle State: ACTIVE
    OCID: ocid1.vaultsecret..<unique_ID>
    ---
    Name: DataScienceSecret_fcacaa
    Lifecycle State: ACTIVE
    OCID: ocid1.vaultsecret..<unique_ID>
    ---


**List Keys**

The ``list_keys`` method of the ``KmsManagementClient`` object provide access returns a list of keys in a specific vault. It returns a ``Response`` object and the ``data`` attribute in that object is a list of ``KeySummary`` objects.

The ``KeySummary`` class has the following attributes:

* ``compartment_id``: OCID of the compartment that the key belongs to.
* ``defined_tags``: Oracle defined tags.
* ``display_name``: Name of the key.
* ``freeform_tags``: User-defined tags. 
* ``id``: OCID of the key.
* ``lifecycle_state``: The lifecycle state such as ENABLED. 
* ``time_created``: Timestamp of when the key was created. 
* ``vault_id``: OCID of the vault that holds the key.

Note, the ``KeySummary`` object does not contain the AES key. When a secret is returned that was encrypted with a key it will automatiacally be decrypted. The most common use-case for a data scientist is to list keys to get the OCID of a desired key but not to interact directly with the key.

The following code uses some of the above attributes to provide details on the keys in a given vault.

.. code-block:: python3

    # Get a list of keys and print some information about each one
    key_list = KmsManagementClient(config, service_endpoint=vault.management_endpoint).list_keys(
                   compartment_id=compartment_id).data
    for key in key_list:
        print("Name: {}\nLifecycle State: {}\nOCID: {}\n---".format(
            key.display_name, key.lifecycle_state,key.id))


.. parsed-literal::

    Name: DataScienceKey_1ddde6
    Lifecycle State: ENABLED
    OCID: ocid1.key..<unique_ID>
    ---

**List Vaults**

The ``list_vaults`` method of the ``KmsVaultClient`` object returns a list of all the vaults in a specific compartment. It returns a ``Response`` object and the ``data`` attribute in that object is a list of ``VaultSummary`` objects.

The ``VaultSummary`` class has the following attributes: 

* ``compartment_id``: OCID of the compartment that the key belongs to. 
* ``crypto_endpoint``: The end-point for encryption and decryption. 
* ``defined_tags``: Oracle defined tags. 
* ``display_name``: Name of the key. 
* ``freeform_tags``: User-defined tags. 
* ``id``: OCID of the vault. 
* ``lifecycle_state``: The lifecycle state, such as ACTIVE. 
* ``time_created``: Timestamp of when the key was created. 
* ``management_endpoint``: Endpoint for managing the vault. 
* ``vault_type``: The oci.key_management.models.Vault type. For example, DEFAULT.

The following code uses some of the above attributes to provide details on the vaults in a given compartment.

.. code-block:: python3

    # Get a list of vaults and print some information about each one.
    vault_list = KmsVaultClient(config).list_vaults(compartment_id=compartment_id).data
    for vault_key in vault_list:
        print("Name: {}\nLifecycle State: {}\nOCID: {}\n---".format(
            vault_key.display_name, vault_key.lifecycle_state,vault_key.id))



.. parsed-literal::

    Name: DataScienceVault_594c0f
    Lifecycle State: ACTIVE
    OCID: ocid1.vault..<unique_ID>
    ---
    Name: DataScienceVault_a10ee1
    Lifecycle State: DELETED
    OCID: ocid1.vault..<unique_ID>
    ---
    Name: DataScienceVault_0cbf46
    Lifecycle State: ACTIVE
    OCID: ocid1.vault..<unique_ID>
    ---
    Name: shay_test
    Lifecycle State: ACTIVE
    OCID: ocid1.vault..<unique_ID>
    ---


**Deletion**

Vaults, keys, and secrets cannot be deleted immediately. They are marked as pending deletion. By default, they are deleted 30 days after they request for deletion. The length of time before deletion is configurable.

**Delete a Secret**

The ``schedule_secret_deletion`` method of the ``VaultsClient`` class is used to delete a secret. It requires the secret’s OCID and a ``ScheduleSecretDeletionDetails`` object. The ``ScheduleSecretDeletionDetails`` provides details about when the secret is deleted.

The ``schedule_secret_deletion`` method returns a ``Response`` object that has information about the deletion process. If the key has already been marked for deletion, a ``ServiceError`` occurs with information about the key.

.. code-block:: python3

    VaultsClient(config).schedule_secret_deletion(secret_id, ScheduleSecretDeletionDetails())

**Delete a Key**

The ``schedule_key_deletion`` method of the ``KmsManagementClient`` class is used to delete a key. It requires the key’s OCID and a ``ScheduleKeyDeletionDetails`` object. The ``ScheduleKeyDeletionDetails`` provides details about when the key is deleted.

The ``schedule_key_deletion`` method returns a ``Response`` object that has information about the deletion process. If the key has already been marked for deletion, a ``ServiceError`` occurs.

Note that secrets are encrypted with a key. If that key is deleted, then the secret cannot be decrypted.

.. code-block:: python3

    KmsManagementClient(config, service_endpoint=vault.management_endpoint).schedule_key_deletion(key_id, ScheduleKeyDeletionDetails())

**Delete a Vault**

The ``schedule_vault_deletion`` method of the ``KmsVaultClient`` class is used to delete a vault. It requires the vault’s OCID and a ``ScheduleVaultDeletionDetails`` object. The ``ScheduleVaultDeletionDetails`` provides details about when the vault
is deleted.

The ``schedule_vault_deletion`` method returns a ``Response`` object that has information about the deletion process. If the vault has already been marked for deletion, then a ``ServiceError`` occurs.

Note that keys and secrets are associated with vaults. If a vault is deleted, then all the keys and secrets in that vault are deleted.

.. code-block:: python3

    try:

Note, the ``KeySummary`` object does not contain the AES key. When a secret is returned that was encrypted with a key it will automatiacally be decrypted. The most common use-case for a data scientist is to list keys to get the OCID of a desired key but not to interact directly with the key.

The following code uses some of the above attributes to provide details on the keys in a given vault.

.. code-block:: python3

    # Get a list of keys and print some information about each one
    key_list = KmsManagementClient(config, service_endpoint=vault.management_endpoint).list_keys(
                   compartment_id=compartment_id).data
    for key in key_list:
        print("Name: {}\nLifecycle State: {}\nOCID: {}\n---".format(
            key.display_name, key.lifecycle_state,key.id))

