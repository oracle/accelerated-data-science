=======================================================================
Marketplace Operator: Setup Feature Store into an existing OKE cluster 
=======================================================================

The Feature store marketplace operator leverages your current infrastructure to setup feature store. It will help in setting up the feature store API server in your exisiting OKE cluster and MySQL database.


**Installing the Feature Store Marketplace Operator**

The Feature Store Marketplace Operator can be installed from PyPi using the following command.

.. code-block:: bash

    python3 -m pip install "https://github.com/oracle/accelerated-data-science.git@feature-store-marketplace[feature-store-marketplace]"


After that, the Operator is ready to go!

**Configuration**

After having set up ads opctl on your desired machine using ads opctl configure, you are ready to begin setting up feature store. At a bare minimum, you will need to provide the following details about your infrastructure:

- Path to the OCIR repository where feature store container images will be cloned
- Compartment ID where feature store will be setup
- :ref:`Database configuration`
- App name to use for Helm
- Namespace to use in the Kubernetes cluster
- Version of feature store stack to install

Optionally you can also specify details for API Gateway setup for feature store to enable authentication and authorization via OCI IAM:
 - Tenancy OCID
 - Region of deployment
 - User group OCIDs authorized to use feature store
 - OCID of resource manager stack to use for API Gateway deployment

These details can be easily configured in an interactive manner by running the command ``ads operator init --type feature_store_marketplace``

**Prerequisites for running the operator**

Before running the operator we need to also configure the following additional requirements:

1. Helm: Helm is required to be installed on the machine for deploying feature store helm chart to the Kubernetes cluster. Ref: `Installing Helm   <https://helm.sh/docs/intro/install/>`_
2. Kubectl: Kubectl is required to be installed to deploy the helm chart to the cluster. Ref: `Installing Kubectl <https://kubernetes.io/docs/tasks/tools/>`_
3. :ref:`Policies`: Required policies for API server
4. `Setup cluster access locally: <https://docs.oracle.com/en-us/iaas/Content/ContEng/Tasks/contengdownloadkubeconfigfile.htm#:~:text=Under%20Containers%20%26%20Artifacts%2C%20click%20Kubernetes,shows%20details%20of%20the%20cluster>`_


**Run**

After the feature_store_marketplace.yaml is written using the init step above we can simply start the deployment using:

.. code-block:: bash

    ads operator run -f feature_store_marketplace.yaml


**Common Issues**
 -- TODO --

.. _Policies:

**Policies**

Policies required by feature store API server:
.. code-block:: text

    allow dynamic-group <feature-store-dynamic-group> to read compartments in tenancy

    allow dynamic-group <feature-store-dynamic-group> to manage data-catalog-family in tenancy

    allow dynamic-group <feature-store-dynamic-group> to insect data-science-models in tenancy

Here ``feature-store-dynamic-group`` is the dynamic group corresponding to the instances of the OKE nodepool where the server is deployed. `Dynamic groups <https://docs.oracle.com/en-us/iaas/Content/Identity/Tasks/callingservicesfrominstances.htm#:~:text=Dynamic%20groups%20allow%20you%20to,against%20Oracle%20Cloud%20Infrastructure%20services.>`_ 


**Database configuration**

Feature store can be configured to utilise your existing MySQL database. It supports two types of authentication:

1.  Basic (Not recommended): Password is stored as plaintext in the API server
2.  Vault (Recommended): Password is stored in an encrypted format inside `OCI Vault <https://docs.oracle.com/en-us/iaas/Content/KeyManagement/Concepts/keyoverview.htm>`_. 

Storing password in Vault:

1. (Optional) Create a new Vault
2. (Required) Create a secret of plain-text type containing the db password
3. (Required) Additional policies for feature store api dynamic group to allow reading the secret from Vault:
    - ``Allow dynamic-group <feature-store-dynamic-group> to use secret-family in tenancy``

Here ``feature-store-dynamic-group`` is the dynamic group corresponding to the instances of the OKE nodepool where the server is deployed. `Dynamic groups <https://docs.oracle.com/en-us/iaas/Content/Identity/Tasks/callingservicesfrominstances.htm#:~:text=Dynamic%20groups%20allow%20you%20to,against%20Oracle%20Cloud%20Infrastructure%20services.>`_ 


