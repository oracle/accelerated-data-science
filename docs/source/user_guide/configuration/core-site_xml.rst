.. _configuration-core_site_xml:

``core-site.xml``
*****************

The ``core-site.xml`` is used to configure connections to Data Flow. This file can be configured manually or with the aid of the ``odsc`` command-line tool. The best practice is to use the ``odsc core-site config`` command-line tool when you want to connect to Data Flow. It gathers information about your environment and uses that to build the file.

The ``odsc core-site config`` command-line tool has no required parameters. Default values are used or values are taken from your notebook session environment and OCI configuration file. Below is a discussion of common parameters that you may need to override.

The ``--authentication`` option sets the authentication mode. It supports resource principal and API keys. The preferred method for authentication is resource principal and this is sent with ``--authentication resource_principal``. If you want to use API keys then used the option ``--authentication api_key``. If the ``--authentication`` is not specified, API keys will be used. When API keys are used, information from the OCI configuration file is used to create the ``core-site.xml`` file.

The Object Storage and the Data Flow are regional services. By default, the region is set to the region that your notebook session is in. This information is taken from the environment variable ``NB_REGION``. Use the ``--region`` option to override this behavior.

The default location of the ``core-site.xml`` file is in the ``~/spark_conf_dir`` directory, as defined in the ``SPARK_CONF_DIR`` environment variable. Use the ``--output`` option to define the directory where the file is to be written.

``odsc`` Command-line
=====================

The ``odsc core-site config`` command-line tool is ideal for setting up the ``core-site.xml`` file as it gathers information about your environment and uses that to build the file.

You will need to determine what settings are appropriate for your configuration. However, the following will work for most configurations.

.. code-block:: bash

    odsc core-site config --authentication resource_principal

If the option ``--authentication api_key`` is used, it will extract information from the OCI configuration file that is stored in ``~/.oci/config``.

For details on the command-line option use the command:

.. code-block:: bash

   odsc core-site config --help

Manual
======

The ``odsc`` command-line tool is the preferred method for configuring the ``core-site.xml`` file. However, if you are not in a notebook session or if you have special requirements, you may need to manually configure the file. This section will guide you through the steps.

The ``core-site.xml`` file has the following format. The name of the parameter goes in between the ``<name> </name>`` tags and the value goes in between the ``<value> </value>`` tags. Each parameter is in between the ``<property> </property>`` tags.

.. code-block:: xml

   <?xml version="1.0"?>
   <configuration>
      <property>
         <name>NAME_1</name>
         <value>VALUE_1</value>
      </property>
      <property>
         <name>NAME_2</name>
         <value>VALUE_2</value>
      </property>
   </configuration>

The ``fs.oci.client.hostname`` needs to be specified. It is the address of Object Storage. For example, ``https://objectstorage.us-ashburn-1.oraclecloud.com`` You have to replace ``us-ashburn-1`` with the region you are in.

Depending on the authentication method that is to be used there are additional parameters that need to be set. See the following sections for guidance.

Resource Principals
-------------------

Update the ``core-site.xml``  file parameters to use resource principal to authenticate:

* ``fs.oci.client.custom.authenticator``: Set the value to ``com.oracle.bmc.hdfs.auth.ResourcePrincipalsCustomAuthenticator``.

The following example ``core-site.xml`` file illustrates using resource principals for authentication to Object Storage:

.. code-block:: xml

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

For details, see `HDFS connector for Object Storage using a resource principal for authentication <https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/hdfsconnector.htm#hdfs_using_resource_principals_for_authentication>`_.

API Keys
--------

Update the ``core-site.xml`` file parameters to use API keys to authenticate:

* ``fs.oci.client.auth.fingerprint``: Fingerprint for the key pair.
* ``fs.oci.client.auth.passphrase``: An optional password phrase if the PEM key is encrypted.
* ``fs.oci.client.auth.pemfilepath``: The fully qualified file name of the private key used for authentication.
* ``fs.oci.client.auth.tenantId``: OCID of your tenancy.
* ``fs.oci.client.auth.userId``: Your user OCID.

The values of these parameters are found in the OCI configuration file.

.. code-block:: xml

   <?xml version="1.0"?>
   <configuration>
      <property>
         <name>fs.oci.client.hostname</name>
         <value>https://objectstorage.us-ashburn-1.oraclecloud.com</value>
      </property>
      <property>
         <name>fs.oci.client.auth.tenantId</name>
         <value>ocid1.tenancy.oc1..<unique_id></value>
      </property>
      <property>
         <name>fs.oci.client.auth.userId</name>
         <value>ocid1.user.oc1..<unique_id></value>
      </property>
      <property>
         <name>fs.oci.client.auth.fingerprint</name>
         <value>01:23:45:67:89:ab:cd:ef:01:23:45:67:89:ab:cd:ef</value>
      </property>
      <property>
         <name>fs.oci.client.auth.pemfilepath</name>
         <value>/home/datascience/.oci/<filename>.pem</value>
      </property>
   </configuration>

