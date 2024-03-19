=================================
Feature Store Deployment Operator (Recommended)
=================================

The Feature Store marketplace operator uses your current infrastructure to set up a Feature Store. It helps in setting up the Feature Store API server in your exisiting OKE cluster and MySQL database.


Installing the Feature Store Marketplace Operator
-------------------------------------------------

The Feature Store Marketplace Operator can be installed from PyPi using the following command.

.. code-block:: bash

    python3 -m pip install oracle-ads[feature-store-marketplace]


After that, the Operator is ready to go!

Configuration
-------------

After having set up `ads opctl` on your desired machine using ``ads opctl configure``, you are ready to begin setting up Feature Store. At a minimum, you need to provide the following details about your infrastructure:

- The path to the OCIR repository where Feature Store container images are cloned.
- The compartment ID where Feature Store is set up.
- :ref:`Database configuration details <Database configuration>`
- The app name to use for Helm.
- The namespace to use in the Kubernetes cluster.
- The version of the Feature Store stack to install.

Optionally you can specify details for the  API Gateway setup for Feature Store to enable authentication and authorization via OCI IAM:
 - Tenancy OCID
 - Region of deployment
 - User group OCIDs authorized to use Feature Store
 - OCID of resource manager stack to use for API Gateway deployment

These details can be easily configured in an interactive manner by running the command

.. code-block:: bash

    ads operator init --type feature_store_marketplace

Prerequisites for running the operator
----------------------------------------

Before running the operator you need to configure the following requirements:

1. Helm: Helm is required to be installed on the machine for deploying Feature Store helm chart to the Kubernetes cluster. Ref: `Installing Helm   <https://helm.sh/docs/intro/install/>`_
2. Kubectl: Kubectl is required to be installed to deploy the helm chart to the cluster. Ref: `Installing Kubectl <https://kubernetes.io/docs/tasks/tools/>`_
3. :ref:`Policies`: Required policies for API server and user running the operator.
4. `Setup cluster access locally <https://docs.oracle.com/en-us/iaas/Content/ContEng/Tasks/contengdownloadkubeconfigfile.htm#:~:text=Under%20Containers%20%26%20Artifacts%2C%20click%20Kubernetes,shows%20details%20of%20the%20cluster>`_


Run
----

After the ``feature_store_marketplace.yaml`` is written using the init step above, you can start the deployment using:

.. important::

 The installation will happen by default in ``default`` namespace. Ensure that NLB (Network Load Balancer) limit does not exceed the maximum limit.

.. code-block:: bash

    ads operator run -f feature_store_marketplace.yaml -b marketplace
