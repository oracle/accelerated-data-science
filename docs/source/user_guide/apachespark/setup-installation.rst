Setup and Installation
***********************

Notebook Session Development
=============================

Follow these set up instructions to submit Spark Jobs to Data Flow from an OCI Notebook Session.

Pyspark Environment
-------------------

To setup PySpark environment, install one of the PySpark conda environments from the ``Environment Explorer``

Example - 

.. code-block:: shell

    odsc conda install -s pyspark30_p37_cpu_v5

Find the information about the latest pyspark conda environment `here <https://docs.oracle.com/en-us/iaas/data-science/using/conda-pyspark-fam.htm>`_

Activate the conda environment to upgrade to the latest ``oracle-ads``

.. code-block:: shell

    conda activate /home/datascience/conda/pyspark30_p37_cpu_v5
    pip install oracle-ads[data_science, data, opctl] --upgrade


Configuring core-site.xml
-------------------------

When the conda environment is installed, a templated version of `core-site.xml` is also installed. You can update the `core-site.xml` file using an automated configuration or manually.

**Authentication with Resource Principals**

Authentication to Object Storage can be done with a resource principal.

For automated configuration, run the following command in a terminal 

.. code-block:: bash

    odsc core-site config -a resource_principal 
    
This command will populate the file ``~/spark_conf_dir/core-site.xml`` with the values needed to connect to Object Storage.

The following command line options are available:

- `-a`, `--authentication` Authentication mode. Supports `resource_principal` and `api_key` (default).
- `-r`, `--region` Name of the region.
- `-o`, `--overwrite` Overwrite `core-site.xml`.
- `-O`, `--output` Output path for `core-site.xml`.
- `-q`, `--quiet` Suppress non-error output.
- `-h`, `--help` Show help message and exit.

To manually configure the ``core-site.xml`` file, you edit the file, and then specify these values:

``fs.oci.client.hostname``: The Object Storage endpoint for your region. See `https://docs.oracle.com/iaas/api/#/en/objectstorage/20160918/` for available endpoints.

``fs.oci.client.custom.authenticator``: Set the value to `com.oracle.bmc.hdfs.auth.ResourcePrincipalsCustomAuthenticator`.

When using resource principals, these properties don't need to be configured:

- ``fs.oci.client.auth.tenantId``
- ``fs.oci.client.auth.userId``
- ``fs.oci.client.auth.fingerprint``
- ``fs.oci.client.auth.pemfilepath``

The following example `core-site.xml` file illustrates using resource principals for authentication to access Object Storage:

::

  <?xml version="1.0"?>
  <configuration>
    <property>
      <name>fs.oci.client.hostname</name>
      <value>https://objectstorage.us-ashburn-1.oraclecloud.com</value>
    </property>
    <property>
      <name>fs.oci.client.custom.authenticator</name>
      <value>com.oracle.bmc.hdfs.auth.ResourcePrincipalsCustomAuthenticator</value>
    </property>
  </configuration>

For details, see `HDFS connector for Object Storage #using resource principals for authentication <https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/hdfsconnector.htm#hdfs_using_resource_principals_for_authentication>`_.



**Authentication with API Keys**

When using authentication with **API keys**, the `core-site.xml` file is be updated in two ways, automated or manual configuration.

For automated configuration, you use the `odsc` command line tool. With an OCI configuration file, you can run 

.. code-block:: bash

    odsc core-site config -o

By default, this command uses the OCI configuration file stored in ``~/.oci/config``, automatically populates the ``core-site.xml`` file,
and then saves it to ``~/spark_conf_dir/core-site.xml``.

The following command line options are available:

- `-a`, `--authentication` Authentication mode. Supports `resource_principal` and `api_key` (default).
- `-c`, `--configuration` Path to the OCI configuration file.
- `-p`, `--profile` Name of the profile.
- `-r`, `--region` Name of the region.
- `-o`, `--overwrite` Overwrite `core-site.xml`.
- `-O`, `--output` Output path for `core-site.xml`.
- `-q`, `--quiet` Suppress non-error output.
- `-h, \--help` Show help message and exit.

To manually configure the ``core-site.xml`` file, you must specify these parameters:

``fs.oci.client.hostname``:  Address of Object Storage. For example, `https://objectstorage.us-ashburn-1.oraclecloud.com`. You must replace us-ashburn-1 with the region you are in.
``fs.oci.client.auth.tenantId``: OCID of your tenancy.
``fs.oci.client.auth.userId``: Your user OCID.
``fs.oci.client.auth.fingerprint``: Fingerprint for the key pair.
``fs.oci.client.auth.pemfilepath``: The fully qualified file name of the private key used for authentication.

The values of these parameters are found in the OCI configuration file.



Local Development
==================

Follow these set up instructions to submit Spark Jobs to Data Flow from your local machine.

PySpark Environment
--------------------

**Prerequisite**

    You have completed :doc:`Local Development Environment Setup<../cli/opctl/local-development-setup>`

Use ``ADS CLI`` to setup a PySpark conda environment. Currently, the ADS CLI only supports fetching conda packs ``published`` by you. If you haven't already published a conda pack, you can create one using ``ADS CLI``

To install from your published environment source - 

.. code-block:: shell

    ads conda install oci://mybucket@mynamespace/path/to/pyspark/env

To create a conda pack for your local use - 

.. code-block:: shell

    cat <<EOF> pyspark.yaml
        
        dependencies:
            - pyspark
            - pip
            - pip:
                - oracle-ads
        name: pysparkenv
    EOF

.. code-block:: shell

    ads create -f pyspark.yaml

.. code-block:: shell

    ads publish -s pysparkenv


Developing in Visual Studio Code
--------------------------------

**Prerequisites**

1. Setup Visual Studio Code development environment by following steps from :doc:`Local Development Environment Setup<../cli/opctl/local-development-setup>`
2. ``ads conda install <oci uri of pyspark conda environment>``. Currently, we cannot access service pack directly. You can instead publish a pyspark service pack to your object storage and use the URI for the pack in OCI Object Storage.

Once the development environment is setup, you could write your code and run it from the terminal of the Visual Studio Code.

``core-site.xml`` is setup automatically when you install a pyspark conda pack.


Logging From DataFlow
=====================

If using the ADS Python SDK, 

To create and run a Data Flow application, you must specify a 
compartment and a bucket for storing logs under the same 
compartment:

.. code-block:: python

    compartment_id = "<compartment_id>"
    logs_bucket_uri = "<logs_bucket_uri>"

Ensure that you set up the correct policies. For instance, for
Data Flow to access logs bucket, use a policy like:

::

   ALLOW SERVICE dataflow TO READ objects IN tenancy WHERE target.bucket.name='dataflow-logs'

For more information, see the `Data Flow documentation <https://docs.oracle.com/en-us/iaas/data-flow/using/dfs_getting_started.htm#set_up_admin>`__.

