Authentication
==============

When you are working within a notebook session, you are operating as the ``datascience`` Linux user. This user does not have an OCI Identity and Access Management (IAM) identity, so it has no access to the Oracle Cloud Infrastructure API. Oracle Cloud Infrastructure resources include Data Science projects, models, jobs, model deployment, and the resources of other OCI services, such as Object Storage, Functions, Vault, Data Flow, and so on. To access these resources, you must use one of the two provided authentication approaches:


1. Authenticating Using Resource Principals
-------------------------------------------

**Prerequisite**

* You are operating within a OCI service that has resource principal based authentication configured
* You have setup the required policies allowing the ``resourcetype`` within which you are operating to use/manage the target OCI resources.


This is the generally preferred way to authenticate with an OCI service. A resource principal is a feature of IAM that enables resources to be authorized principal actors that can perform actions on service resources. Each resource has its own identity, and it authenticates using the certificates that are added to it. These certificates are automatically created, assigned to resources, and rotated avoiding the need for you to upload credentials to your notebook session.

Data Science enables you to authenticate using your notebook session's resource principal to access other OCI resources. When compared to using the OCI configuration and key files approach, using resource principals provides a more secure and easy way to authenticate to the OCI APIs.

You can choose to use the resource principal to authenticate while using the Accelerated Data Science (ADS) SDK by running ``ads.set_auth(auth='resource_principal')`` in a notebook cell. For example:

.. code-block:: python

  import ads
  ads.set_auth(auth='resource_principal')
  compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
  pc = ProjectCatalog(compartment_id=compartment_id)
  pc.list_projects()


2. Authenticating Using API Keys
--------------------------------

**Prerequisite**

* You have setup api keys as per the instruction `here <https://docs.oracle.com/en-us/iaas/Content/API/Concepts/apisigningkey.htm>`_

Use API Key setup when you are working from a local workstation or on platform which does not support resource principals.

This is the default method of authentication. You can also authenticate as your own personal IAM user by creating or uploading OCI configuration and API key files inside your notebook session environment. The OCI configuration file contains the necessary credentials to authenticate your user against the model catalog and other OCI services like Object Storage. The example notebook, `api_keys.ipynb` demonstrates how to create these files.

You can follow the steps in `api_keys.ipynb <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/master/notebook_examples/api_keys.ipynb>`_ for step by step instruction on setting up API Keys.

.. note::
   If you already have an OCI configuration file (``config``) and associated keys, you can upload them directly to the ``/home/datascience/.oci`` directory using the JupyterLab **Upload Files** or the drag-and-drop option.


3. Authenticating Using Instance Principals
-------------------------------------------

**Prerequisite**

* You are operating within an OCI compute instance
* You have created a Dynamic Group with Matching Rules to include your compute instances, and you have authored policies allowing this Dynamic Group to perform actions within your tenancy

For more information on Instance Principals, see `Calling Services from an Instance <https://docs.oracle.com/iaas/Content/Identity/Tasks/callingservicesfrominstances.htm>`_.

You can choose to use the instance principal to authenticate while using the Accelerated Data Science (ADS) SDK by running ``ads.set_auth(auth='instance_principal')``. For example:

.. code-block:: python

  import ads
  ads.set_auth(auth='instance_principal')
  mc = ModelCatalog(compartment_id="<compartment_id>")
  mc.list_models()


4. Overriding Defaults
----------------------

The default authentication that is used by ADS is set with the ``set_auth()`` method. However, each relevant ADS method has an optional parameter to specify the authentication method to use. The most common use case for this is when you have different permissions in different API keys or there are differences between the permissions granted in the resource principals and your API keys.

By default, ADS uses API keys to sign requests to OCI resources. The ``set_auth()`` method is used to explicitly set a default signing method. This method accepts one of three strings ``"api_key"``, ``"resource_principal"``, or ``instance_principal``.

The ``~/.oci/config`` configuration allow for multiple configurations to be stored in the same file. The ``set_auth()`` method takes is ``oci_config_location`` parameter that specifies the location of the configuration, and the default is ``"~/.oci/config"``. Each configuration is called a profile, and the default profile is ``DEFAULT``. The ``set_auth()`` method takes in a parameter ``profile``. It specifies which profile in the ``~/.oci/config`` configuration file to use. In this context, the ``profile`` parameter is only used when API keys are being used. If no value for ``profile`` is specified, then the ``DEFAULT`` profile section is used.

.. code-block:: python

  import ads
  import oci

  ads.set_auth("api_key") # default signer is set to API Keys
  ads.set_auth("api_key", profile = "TEST") # default signer is set to API Keys and to use TEST profile
  ads.set_auth("api_key", oci_config_location = "~/.test_oci/config") # default signer is set to API Keys and to use non-default oci_config_location
  ads.set_auth("resource_principal")  # default signer is set to resource principal authentication
  ads.set_auth("instance_principal")  # default signer is set to instance principal authentication

  singer = oci.auth.signers.ResourcePrincipalsFederationSigner()
  ads.set_auth(config={}, singer=signer) # default signer is set to ResourcePrincipalsFederationSigner

  signer_callable = oci.auth.signers.ResourcePrincipalsFederationSigner
  ads.set_auth(signer_callable=signer_callable) #  default signer is set ResourcePrincipalsFederationSigner callable

The ``auth`` module has helper functions that return a signer which is used for authentication. The ``api_keys()`` method returns a signer that uses the API keys in the ``.oci`` configuration directory. There are optional parameters to specify the location of the API keys and the profile section. The ``resource_principal()`` method returns a signer that uses resource principals. The method ``default_signer()`` returns either a signer for API Keys or resource principals depending on the defaults that have been set. The ``set_auth()`` method determines which signer type is the default. If nothing is set then API keys are the default.

Additional signers may be provided by running ``set_auth()`` with ``signer`` or ``signer_callable`` with optional ``signer_kwargs`` parameters. You can find the list of additional signers `here <https://docs.oracle.com/iaas/tools/python/latest/api/signing.html>`_.

.. code-block:: python

  from ads.common import auth as authutil
  from ads.common import oci_client as oc

  # Example 1: Create Object Storage client with the default signer.
  auth = authutil.default_signer()
  oc.OCIClientFactory(**auth).object_storage

  # Example 2: Create Object Storage client with timeout set to 6000 using resource principal authentication.
  auth = authutil.resource_principal({"timeout": 6000})
  oc.OCIClientFactory(**auth).object_storage

  # Example 3: Create Object Storage client with timeout set to 6000 using API Key authentication.
  auth = authutil.api_keys(oci_config="/home/datascience/.oci/config", profile="TEST", kwargs={"timeout": 6000})
  oc.OCIClientFactory(**auth).object_storage


In the this example, the default authentication uses API keys specified with the ``set_auth`` method. However, since the ``os_auth`` is specified to use resource principals, the notebook session uses the resource principal to access OCI Object Store.

.. code-block:: python

  set_auth("api_key") # default signer is set to api_key
  os_auth = authutil.resource_principal() # use resource principal to as the preferred way to access object store


More signers can be created using the ``create_signer()`` method. With the ``auth_type`` parameter set to ``instance_principal``, the method will return a signer that uses instance principals. For other signers there are ``signer`` or ``signer_callable`` parameters. Here are examples:

.. code-block:: python

  import ads
  import oci

  # Example 1. Create signer that uses instance principals
  auth = ads.auth.create_signer("instance_principal")

  # Example 2. Provide a ResourcePrincipalsFederationSigner object
  singer = oci.auth.signers.ResourcePrincipalsFederationSigner()
  auth = ads.auth.create_signer(config={}, singer=signer)

  # Example 3. Create signer that uses instance principals with log requests enabled
  signer_callable = oci.auth.signers.InstancePrincipalsSecurityTokenSigner
  signer_kwargs = dict(log_requests=True) # will log the request url and response data when retrieving
  auth = ads.auth.create_signer(signer_callable=signer_callable, signer_kwargs=signer_kwargs)