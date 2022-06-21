.. _BDS Connect:

Connect
*******

Notebook Session
================

Notebook sessions require a conda environment that has the BDS module of ADS installed.

Using the Vault
---------------

The preferred method to connect to a BDS cluster is to use the ``BDSSecretKeeper`` class. This allows you to store the BDS credentials in
the vault and not the notebook. It also provides a greater level of access control to the secrets and allows for credential rotation
without breaking connections from various sources. 

.. include:: _template/hive_connect_with_vault.rst

Without Using the Vault
-----------------------

BDS requires a Kerberos ticket to authenticate to the service. The preferred method is to use the vault and ``BDSSecretKeeper``
because it is more secure, and prevents private information from being stored in a notebook. However, if this is not possible,
you can use the ``refresh_ticket()`` method to manually create the Kerberos ticket. This method requires the following parameters:

* ``kerb5_path``: The path to the ``krb5.conf`` file. You can copy this file  from the master node of the BDS cluster located in ``/etc/krb5.conf``.
* ``keytab_path``: The path to the principal's ``keytab`` file. You can download this file from the master node on the BDS cluster.
* ``principal``: The unique identity to that Kerberos can assign tickets to.

.. include:: _template/hive_connect_without_vault.rst

Jobs
====

A job requires a conda environment that has the BDS module of ADS installed. It also requires secrets and configuration information that can be used to obtain a Kerberos ticket for authentication. You must copy the ``keytab`` and ``krb5.conf`` files to the jobs instance and can be copied as part of the job. We recommend that you save them into the vault then use ``BDSSecretKeeper`` to access them. This is secure because the vault provides access control and allows for key rotation without breaking exiting jobs. You can use the notebook to load configuration parameters like ``hdfs_host``, ``hdfs_port``, ``hive_host``, ``hive_port``, and so on. The ``keytab`` and ``krb5.conf`` files are securely loaded from the vault then saved in the jobs instance. The ``krbcontext()`` method is then used to create the Kerberos ticket. Once the ticket is created, you can query BDS.
