===========================
AI Quick Actions API Server
===========================

AI Quick Actions is accessible through the Python SDK APIs and CLI. If the CLI or Python SDK doesn't work for you, you can host the Aqua API server and integrate with it. We also provide you with the OpenAPI 3.0 specification so that you can autogenerate client bindings for the language of your choice.

**Prerequisite**

1. Install oracle-ads - ``pip install "oracle-ads[aquaapi]"``
2. Set up AI Quick Actions `policies <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/ai-quick-actions/policies/README.md>`_

API Specification
=================

Access API specification from `aqua_spec.yaml <https://github.com/oracle/accelerated-data-science/blob/main/ads/aqua/server/aqua_spec.yaml>`_ 


Configuration
=============

The preferred way to set up the configuration is to create a file named ``.env`` and set up your preferences. Here is a sample content of ``.env``:

.. code-block:: shell

    OCI_IAM_TYPE=security_token
    OCI_CONFIG_PROFILE=aqua
    HF_TOKEN=<your token>

Authentication
--------------

AI Quick Actions will need to connect to OCI services to accomplish different functionalities. You can use api_key, security_token, resource_principal, instance_principal, etc. 

You can set up the preferred authentication mechanism through the following environment variables - 

.. code-block:: shell

    # set this to api_key/resource_principal/instance_principal/security_token
    OCI_IAM_TYPE=security_token
    # Optional Profile name
    OCI_CONFIG_PROFILE=<profile-name>

Set up Hugging Face token, if you will be registering the model with the download from Hugging Face option - 

.. code-block:: shell

    HF_TOKEN=<your token>


Default Settings
----------------

You can set up the following default values to avoid having to input them during API calls - 

.. code-block:: shell

    NB_SESSION_COMPARTMENT_OCID=ocid1.compartment...<UNIQUEID>
    PROJECT_OCID=ocid1.datascienceproject.oc1...<UNIQUEID>

    # Optional - If you are on a dev tenancy, you may be restricted from using default network settings. In that case, set up AQUA_JOB_SUBJECT_ID to the preferred subnet ID. This is required only while launching FineTuning jobs
    AQUA_JOB_SUBNET_ID=ocid1.subnet.oc1...<UNIQUEID>

Webserver Settings
------------------

.. code-block:: shell

    # Default value is 8080
    AQUA_PORT=8080
    # Default value is 0.0.0.0
    AQUA_HOST="0.0.0.0"
    # Default value is 0. Set the number of processes you wish to start to handle requests
    AQUA_PROCESS_COUNT=1 
    # If you face CORS related issues while accessing the API, set - 
    AQUA_CORS_ENABLE=1

Starting the server
===================

Once you have the ``.env`` file ready, you can start the server from the same folder as ``.env`` by running - 

.. code-block:: shell

    python -m ads.aqua.server
