====================
Setup Feature Store
====================

Feature store is being provided by OCI as a stack based offering in user's own tenancy via OCI marketplace. It can be configured by the user primarily in two ways:

:doc:`Setup via Feature Store Marketplace Operator (Recommended) <./user_guides.setup.feature_store_operator>`
__________________________________________________________

The feature store marketplace operator can be used to setup the feature store api server in an existing OKE cluster while also utilising an existing MySQL database. It will also help setup authentication and authorization using OCI. For more details, see :doc:`Marketplace operator <./user_guides.setup.feature_store_operator>`. Optionally, we can also setup `Feature Store API Gateway stack <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/feature_store/README.md>`_ for authentication and authorization.

:doc:`Setup via Helm Charts <./user_guides.setup.helm_chart>`
_______________________________

We can manually export images to OCIR using Marketplace UI and then deploy the obtained Helm Chart. Optionally, we can also setup `Feature Store API Gateway stack <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/feature_store/README.md>`_ for authentication and authorization.

.. _Database configuration:

Database configuration
-----------------------

Feature Store can be configured to use your existing MySQL database. It supports two types of authentication:

1.  Basic (Not recommended): The password is stored as plaintext in the API server.
2.  Vault (Recommended): The password is stored in an encrypted format inside `OCI Vault <https://docs.oracle.com/en-us/iaas/Content/KeyManagement/Concepts/keyoverview.htm>`_.

Storing the password in Vault:

1. (Optional) Create a new Vault.
2. (Required) Create a secret of plain-text type containing the db password.
3. (Required) Additional policies for the Feature Store API dynamic group to allow reading the secret from Vault:
    - ``Allow dynamic-group <feature-store-dynamic-group> to use secret-family in tenancy``

Here ``feature-store-dynamic-group`` is the dynamic group corresponding to the instances of the OKE nodepool where the server is deployed.

.. seealso::
    `Dynamic groups <https://docs.oracle.com/en-us/iaas/Content/Identity/Tasks/callingservicesfrominstances.htm#:~:text=Dynamic%20groups%20allow%20you%20to,against%20Oracle%20Cloud%20Infrastructure%20services.>`_


.. _Policies:
Policies
---------

Policies required by the user deploying:

.. code-block:: text

    allow group <user-group> to manage clusters in compartment <compartment-name>
    allow group <user-group> to use repos in compartment <compartment-name>
    allow group <user-group> to manage marketplace-listings in compartment <compartment-name>
    allow group <user-group> to read compartments in compartment <compartment-name>
    allow group <user-group> to manage app-catalog-listing in compartment <compartment-name>
    allow group <user-group> to read object-family in tenancy
    allow group <user-group> to read marketplace-workrequests in compartment FeatureStoreDeployment

The policies required by the Feature Store API server are:

.. code-block:: text

    allow dynamic-group <feature-store-dynamic-group> to read compartments in tenancy

    allow dynamic-group <feature-store-dynamic-group> to manage data-catalog-family in tenancy

    allow dynamic-group <feature-store-dynamic-group> to inspect data-science-models in tenancy


Here ``feature-store-dynamic-group`` is the dynamic group corresponding to the instances of the OKE nodepool where the server is deployed. `Dynamic groups <https://docs.oracle.com/en-us/iaas/Content/Identity/Tasks/callingservicesfrominstances.htm#:~:text=Dynamic%20groups%20allow%20you%20to,against%20Oracle%20Cloud%20Infrastructure%20services.>`_

.. _Known Issues:

Known Issues
-------------

1. Deployment doesn't work in Virtual Nodepool as the Feature Store API server relies on Instance Principal authentication.


.. toctree::
  :maxdepth: 1

  ./user_guides.setup.feature_store_operator
  ./user_guides.setup.helm_chart
