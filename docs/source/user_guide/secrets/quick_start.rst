Quick Start
***********

Auth Tokens
===========

Save Credentials
----------------

.. code-block:: python3

    import ads
    from ads.secrets.auth_token import AuthTokenSecretKeeper

    ads.set_auth('resource_principal') # If using resource principal authentication

    ocid_vault = "ocid1.vault..<unique_ID>"
    ocid_master_key = "ocid1.key..<unique_ID>"
    ocid_mycompartment = "ocid1.compartment..<unique_ID>"

    authtoken2 = AuthTokenSecretKeeper(
                    vault_id=ocid_vault,
                    key_id=ocid_master_key,
                    compartment_id=ocid_mycompartment,
                    auth_token="<your_auth_token>"
                   ).save(
                        "my_xyz_auth_token2",
                        "This is my key for git repo xyz",
                        freeform_tags={"gitrepo":"xyz"}
                    )
    print(authtoken2.secret_id)

``'ocid1.vaultsecret..<unique_ID>'``

Load Credentials
----------------

.. code-block:: python3

    import ads
    from ads.secrets.auth_token import AuthTokenSecretKeeper

    ads.set_auth('resource_principal') # If using resource principal authentication

    with AuthTokenSecretKeeper.load_secret(source="ocid1.vaultsecret..<unique_ID>",
                                   ) as authtoken:
        import os
        print(f"Credentials inside `authtoken` object:  {authtoken}")

``Credentials inside `authtoken` object:  {'auth_token': '<your_auth_token>'}``

Autonomous Database
===================

Save Credentials
----------------

.. code-block:: python3

    import ads
    ads.set_auth('resource_principal') # If using resource principal authentication
    from ads.secrets.adb import ADBSecretKeeper

    connection_parameters={
        "user_name":"admin",
        "password":"<your_password>",
        "service_name":"service_high",
        "wallet_location":"/home/datascience/Wallet_--------.zip"
    }

    ocid_vault = "ocid1.vault..<unique_ID>"
    ocid_master_key = "ocid1.key..<unique_ID>"
    ocid_mycompartment = "ocid1.compartment..<unique_ID>"

    adw_keeper = ADBSecretKeeper(vault_id=ocid_vault,
                                key_id=ocid_master_key,
                                compartment_id=ocid_mycompartment,
                                **connection_parameters)

    # Store the credentials without storing the wallet file
    adw_keeper.save("adw_employee_att2",
                        "My DB credentials",
                        freeform_tags={"schema":"emp"},
                        save_wallet=True
                    )
    print(adw_keeper.secret_id)

``'ocid1.vaultsecret..<unique_ID>'``

Load Credentials
----------------

.. code-block:: python3

    import ads
    ads.set_auth('resource_principal') # If using resource principal authentication
    from ads.secrets.adb import ADBSecretKeeper

    with ADBSecretKeeper.load_secret("ocid1.vaultsecret..<unique_ID>") as adw_creds2:
        import pandas as pd
        df2 = pd.DataFrame.ads.read_sql("select JOBFUNCTION, ATTRITION from ATTRITION_DATA", connection_parameters=adw_creds2)
        print(df2.head(2))

+-+--------------------+----------+
| |         JOBFUNCTION| ATTRITION|
+-+--------------------+----------+
|0|  Product Management|        No|
+-+--------------------+----------+
|1|  Software Developer|        No|
+-+--------------------+----------+

Big Data Service
================

Save Credentials
----------------

.. code-block:: python3

    import ads
    import fsspec
    import os
    
    from ads.secrets.big_data_service import BDSSecretKeeper
    from ads.bds.auth import has_kerberos_ticket, refresh_ticket, krbcontext
    
    ads.set_auth('resource_principal')

    principal = "<your_principal>"
    hdfs_host = "<your_hdfs_host>"
    hive_host = "<your_hive_host>"
    hdfs_port = <your_hdfs_port>
    hive_port = <your_hive_port>
    vault_id = "ocid1.vault..<unique_ID>"
    key_id = "ocid1.key..<unique_ID>"

    secret = BDSSecretKeeper(
                vault_id=vault_id,
                key_id=key_id,
                principal=principal,
                hdfs_host=hdfs_host,
                hive_host=hive_host,
                hdfs_port=hdfs_port,
                hive_port=hive_port,
                keytab_path=keytab_path,
                kerb5_path=kerb5_path
               )

    saved_secret = secret.save(name="your_bds_config_secret_name",
                            description="your bds credentials",
                            freeform_tags={"schema":"emp"},
                            defined_tags={},
                            save_files=True)

Load Credentials
----------------

.. code-block:: python3

    from ads.secrets.big_data_service import BDSSecretKeeper
    from pyhive import hive

    with BDSSecretKeeper.load_secret(saved_secret.secret_id, keytab_dir="~/path/to/save/keytab_file/") as cred:
        with krbcontext(principal=cred["principal"], keytab_path=cred['keytab_path']):
            hive_cursor = hive.connect(host=cred["hive_host"],
                                       port=cred["hive_port"],
                                       auth='KERBEROS',
                                       kerberos_service_name="hive").cursor()

MySQL
=====

Save Credentials
----------------

.. code-block:: python3

    import ads
    from ads.secrets.mysqldb import MySQLDBSecretKeeper

    vault_id = "ocid1.vault..<unique_ID>"
    key_id = "ocid1.key..<unique_ID>"

    ads.set_auth("resource_principal") # If using resource principal for authentication
    connection_parameters={
        "user_name":"<your user name>",
        "password":"<your password>",
        "host":"<db host>",
        "port":"<db port>",
       "database":"<database>",
    }

    mysqldb_keeper = MySQLDBSecretKeeper(vault_id=vault_id,
                                    key_id=key_id,
                                    **connection_parameters)

    mysqldb_keeper.save("mysqldb_employee", "My DB credentials", freeform_tags={"schema":"emp"})
    print(mysqldb_keeper.secret_id) # Prints the secret_id of the stored credentials

``'ocid1.vaultsecret..<unique_ID>'``

Load Credentials
----------------

.. code-block:: python3

    import ads
    from ads.secrets.mysqldb import MySQLDBSecretKeeper
    ads.set_auth('resource_principal') # If using resource principal authentication

    with MySQLDBSecretKeeper.load_secret(source=secret_id) as mysqldb_creds:
        import pandas as pd
        df2 = pd.DataFrame.ads.read_sql("select JOBFUNCTION, ATTRITION from ATTRITION_DATA", connection_parameters=mysqldb_creds)
        print(df2.head(2))

+-+--------------------+----------+
| |         JOBFUNCTION| ATTRITION|
+-+--------------------+----------+
|0|  Product Management|        No|
+-+--------------------+----------+
|1|  Software Developer|        No|
+-+--------------------+----------+

Oracle Database
===============

Save Credentials
----------------

.. code-block:: python3

    import ads
    from ads.secrets.oracledb import OracleDBSecretKeeper

    vault_id = "ocid1.vault..<unique_ID>"
    key_id = "ocid1.key..<unique_ID>"

    ads.set_auth("resource_principal") # If using resource principal for authentication
    connection_parameters={
         "user_name":"<your user name>",
         "password":"<your password>",
         "service_name":"service_name",
         "host":"<db host>",
         "port":"<db port>",
    }

    oracledb_keeper = OracleDBSecretKeeper(vault_id=vault_id,
                                    key_id=key_id,
                                    **connection_parameters)

    oracledb_keeper.save("oracledb_employee", "My DB credentials", freeform_tags={"schema":"emp"})
    print(oracledb_keeper.secret_id) # Prints the secret_id of the stored credentials

``'ocid1.vaultsecret..<unique_ID>'``

Load Credentials
----------------

.. code-block:: python3

    import ads
    ads.set_auth('resource_principal') # If using resource principal authentication
    from ads.secrets.oracledb import OracleDBSecretKeeper

    with OracleDBSecretKeeper.load_secret(source=secret_id) as oracledb_creds:
        import pandas as pd
        df2 = pd.DataFrame.ads.read_sql("select JOBFUNCTION, ATTRITION from ATTRITION_DATA", connection_parameters=oracledb_creds)
        print(df2.head(2))

+-+--------------------+----------+
| |         JOBFUNCTION| ATTRITION|
+-+--------------------+----------+
|0|  Product Management|        No|
+-+--------------------+----------+
|1|  Software Developer|        No|
+-+--------------------+----------+


