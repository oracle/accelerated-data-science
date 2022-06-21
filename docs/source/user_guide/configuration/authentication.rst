.. _configuration-authentication:

Authentication
**************

When you are working within a notebook session, you are operating as the ``datascience`` Linux user. This user does not have an OCI Identity and Access Management (IAM) identity, so it has no access to the Oracle Cloud Infrastructure (OCI) API. Oracle Cloud Infrastructure resources include Data Science projects and models, and the resources of other OCI service, such as Object Storage, Functions, Vault, Data Flow, and so on. To access these resources from the notebook environment, you must use one of the two provided authentication approaches:


Resource Principals
===================

This is the generally preferred way to authenticate with an OCI service. A resource principal is a feature of IAM that enables resources to be authorized principal actors that can perform actions on service resources. Each resource has its own identity, and it authenticates using the certificates that are added to it. These certificates are automatically created, assigned to resources, and rotated avoiding the need for you to upload credentials to your notebook session.

Data Science enables you to authenticate using your notebook session's resource principal to access other OCI resources. When compared to using the OCI configuration and key files approach, using resource principals provides a more secure and easy way to authenticate to the OCI APIs.

Within your notebook session, you can choose to use the resource principal to authenticate while using the Accelerated Data Science (ADS) SDK by running ``ads.set_auth(auth='resource_principal')`` in a notebook cell. For example:

.. code-block:: python3

  import ads 
  ads.set_auth(auth='resource_principal')
  compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
  pc = ProjectCatalog(compartment_id=compartment_id)
  pc.list_projects()


API Keys
========

This is the default method of authentication. You can also authenticate as your own personal IAM user by creating or uploading OCI configuration and API key files inside your notebook session environment. The OCI configuration file contains the necessary credentials to authenticate your user against the model catalog and other OCI services like Object Storage. The example notebook, `api_keys.ipynb` demonstrates how to create these files.

The ``getting-started.ipynb`` notebook in the home directory of the notebook session environment demonstrates all the steps needed to create the configuration file and the keys. Follow the steps in that notebook before importing and using ADS in your notebooks.

.. note::
   If you already have an OCI configuration file (``config``) and associated keys, you can upload them directly to the ``/home/datascience/.oci`` directory using the JupyterLab **Upload Files** or the drag-and-drop option.


Configuration File
==================

The default authentication that is used by ADS is set with the ``set_auth()`` method. However, each relevant ADS method has an optional parameter to specify the authentication method to use. The most common use case for this is when you have different permissions in different API keys or there are differences between the permissions granted in the resource principals and your API keys.

Most ADS methods do not require a signer to be explicitly given. By default, ADS uses the API keys to sign requests to OCI resources. The ``set_auth()`` method is used to explicitly set a default signing method. This method accepts one of two strings ``"api_key"`` or ``"resource_principal"``.

The ``~/.oci/config`` configuration allow for multiple configurations to be stored in the same file. The ``set_auth()`` method takes is ``oci_config_location`` parameter that specifies the location of the configuration, and the default is ``"~/.oci/config"``. Each configuration is called a profile, and the default profile is ``DEFAULT``. The ``set_auth()`` method takes in a parameter ``profile``. It specifies which profile in the ``~/.oci/config`` configuration file to use. In this context, the ``profile`` parameter is only used when API keys are being used. If no value for ``profile`` is specified, then the ``DEFAULT`` profile section is used.

.. code-block:: python3

  ads.set_auth("api_key") # default signer is set to API Keys
  ads.set_auth("api_key", profile = "TEST") # default signer is set to API Keys and to use TEST profile
  ads.set_auth("api_key", oci_config_location = "~/.test_oci/config") # default signer is set to API Keys and to use non-default oci_config_location


The ``authutil`` module has helper functions that return a signer which is used for authentication. The ``api_keys()`` method returns a signer that uses the API keys in the ``.oci`` configuration directory. There are optional parameters to specify the location of the API keys and the profile section. The ``resource_principal()`` method returns a signer that uses resource principals. The method ``default_signer()`` returns either a signer for API Keys or resource principals depending on the defaults that have been set. The ``set_auth()`` method determines which signer type is the default. If nothing is set then API keys are the default.

.. code-block:: python3

  from ads.common import auth as authutil
  from ads.common import oci_client as oc

  # Example 1: Create Object Storage client with  the default signer.
  auth = authutil.default_signer()
  oc.OCIClientFactory(**auth).object_storage

  # Example 2: Create Object Storage client with timeout set to 6000 using resource principal authentication.
  auth = authutil.resource_principal({"timeout": 6000})
  oc.OCIClientFactory(**auth).object_storag

  # Example 3: Create Object Storage client with timeout set to 6000 using API Key authentication.
  auth = authutil.api_keys(oci_config="/home/datascience/.oci/config", profile="TEST", kwargs={"timeout": 6000})
  oc.OCIClientFactory(**auth).object_storage


In the this example, the default authentication uses API keys specified with the ``set_auth`` method. However, since the ``os_auth`` is specified to use resource principals, the notebook session uses the resource principal to access OCI Object Store.

.. code-block:: python3

  set_auth("api_key") # default signer is set to api_key
  os_auth = authutil.resource_principal() # use resource principal to as the preferred way to access object store

