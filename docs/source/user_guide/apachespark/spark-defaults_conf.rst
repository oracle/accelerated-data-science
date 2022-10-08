.. _configuration-spark_defaults_conf:

``spark-defaults.conf``
***********************

The ``spark-defaults.conf`` file is used to define the properties that are used by Spark. This file can be configured manually or with the aid of the ``odsc`` command-line tool. The best practice is to use the ``odsc data-catalog config`` command-line tool when you want to connect to Data Catalog. It gathers information about your environment and uses that to build the file.

The ``odsc data-catalog config`` command-line tool uses the ``--metastore`` option to define the Data Catalog Metastore OCID. There are no required command-line options.  Default values are used or values are taken from your notebook session environment and OCI configuration file. Below is a discussion of common parameters that you may need to override.

The ``--authentication`` option sets the authentication mode. It supports resource principal and API keys. The preferred method for authentication is resource principal and this is sent with ``--authentication resource_principal``. If you want to use API keys then used the option ``--authentication api_key``. If the ``--authentication`` is not specified, API keys will be used. When API keys are used, information from the OCI configuration file is used to create the ``spark-defaults.conf`` file.

The Object Storage and Data Catalog are regional services. By default, the region is set to the region that your notebook session is in. This information is taken from the environment variable ``NB_REGION``. Use the ``--region`` option to override this behavior.

The default location of the ``spark-defaults.conf`` file is in the ``~/spark_conf_dir`` directory, as defined in the ``SPARK_CONF_DIR`` environment variable. Use the ``--output`` option to define the directory where the file is to be written.

``odsc`` Command-line 
=====================

The ``odsc data-catalog config`` command-line tool is ideal for setting up the ``spark-defaults.conf`` file as it gathers information about your environment and uses that to build the file. 

You will need to determine what settings are appropriate for your configuration. However, the following will work for most configurations.

.. code-block:: bash

   odsc data-catalog config --authentication resource_principal

If the option ``--authentication api_key`` is used, it will extract information from the OCI configuration file that is stored in ``~/.oci/config``. Use the ``--config`` option to change the path and the ``--profile`` option to specify what OCI configuration profile will be used. The default profile is ``DEFAULT``.

A default Data Catalog Metastore OCID can be set using the ``--metastore`` option. This value can be overridden at run-time.

.. code-block:: bash

   odsc data-catalog config --authentication resource_principal --metastore <metastore_id>

The ``<metastore_id>`` must be replaced with the OCID for the Data Catalog Metastore that is to be used.

For details on the command-line option use the command:

.. code-block:: bash

   odsc data-catalog config --help

Manual
======

The ``odsc`` command-line tool is the preferred method for configuring the ``spark-defaults.conf`` file. However, if you are not in a notebook session or if you have special requirements, you may need to manually configure the file. This section will guide you through the steps.

When a Data Science Conda environment is installed, it includes a template of the ``spark-defaults.conf`` file. The following sections provide guidance to make the required changes. 

These parameters define the Object Storage address that backs the Data Catalog entry. This is the location of the data warehouse. You also need to define the address of the Data Catalog Metastore.

*  ``spark.hadoop.fs.oci.client.hostname``: Address of Object Storage for the data warehouse.  For example, ``https://objectstorage.us-ashburn-1.oraclecloud.com``.  Replace ``us-ashburn-1`` with the region you are in.
*  ``spark.hadoop.oci.metastore.uris``: The address of Data Catalog Metastore. For example, ``https://datacatalog.us-ashburn-1.oci.oraclecloud.com/`` Replace ``us-ashburn-1`` with the region you are in.

You can set a default metastore with the following parameter. This can be overridden at run time. Setting it is optional.

*  ``spark.hadoop.oracle.dcat.metastore.id``: The OCID of Data Catalog Metastore. For example, ``ocid1.datacatalogmetastore..<unique_id>``

Depending on the authentication method that is to be used there are additional parameters that need to be set. See the following sections for guidance.

Resource Principal
------------------

Update the ``spark-defaults.conf`` file parameters to use resource principal to authenticate:

*  ``spark.hadoop.fs.oci.client.custom.authenticator``: Set the value to ``com.oracle.bmc.hdfs.auth.ResourcePrincipalsCustomAuthenticator``.
*  ``spark.hadoop.oracle.dcat.metastore.client.custom.authentication_provider``: Set the value to ``com.oracle.bmc.hdfs.auth.ResourcePrincipalsCustomAuthenticator``.

API Keys
--------

Update the ``spark-defaults.conf`` file parameters to use API keys to authenticate:

*  ``spark.hadoop.OCI_FINGERPRINT_METADATA``: Fingerprint for the key pair being used.
*  ``spark.hadoop.OCI_PASSPHRASE_METADATA``: Passphrase used for the key if it is encrypted.
*  ``spark.hadoop.OCI_PVT_KEY_FILE_PATH``: The full path and file name of the private key used for authentication.
*  ``spark.hadoop.OCI_REGION_METADATA``: An Oracle Cloud Infrastructure region. Example: ``us-ashburn-1``
*  ``spark.hadoop.OCI_USER_METADATA``: Your user OCID.
*  ``spark.hadoop.fs.oci.client.auth.fingerprint``: Fingerprint for the key pair being used.
*  ``spark.hadoop.fs.oci.client.auth.passphrase``: Passphrase used for the key if it is encrypted.
*  ``spark.hadoop.fs.oci.client.auth.pemfilepath``: The full path and file name of the private key used for authentication.
*  ``spark.hadoop.fs.oci.client.auth.tenantId``: OCID of your tenancy.
*  ``spark.hadoop.fs.oci.client.auth.userId``: Your user OCID.
*  ``spark.hadoop.fs.oci.client.custom.authenticator``: Set the value to ``com.oracle.pic.dcat.metastore.commons.auth.provider.UserPrincipalsCustomAuthenticationDetailsProvider``
*  ``spark.hadoop.spark.hadoop.OCI_TENANT_METADATA``: OCID of your tenancy.

The values of these parameters are found in the OCI configuration file.

