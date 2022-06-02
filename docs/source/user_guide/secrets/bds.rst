.. _secretbds:

Oracle Big Data Service
=======================

Available with ADS v2.5.10 and greater

To connect to Oracle Big Data Service(BDS) you need the following:

- ``principal``: The unique identity to which Kerberos can assign tickets. It will be used to generate the kerberos ticket.
- ``kerb5 config file``: krb5.conf file which can be copied from /etc/krb5.conf from the master node of the BDS cluster. It will be used to generate the kerberos ticket.
- ``keytab file``: The principal's keytab file which can be downloaded from the master node of the BDS cluster. It will be used to generate the kerberos ticket.
- ``hdfs host``: hdfs host name which will be used to connect to the hdfs file system.
- ``hdfs port``: hdfs port which will be used to connect to the hdfs file system.
- ``hive host``: hive host name which will be used to connect to the Hive Server.
- ``hive port``: hive port which will be used to connect to the Hive Server.


The ``BDSSecretKeeper`` class saves the BDS credentials to the OCI Vault service.


Saving Credentials
------------------

**Prerequisites**

- OCID of the vault created in the OCI Console.
- OCID of the master key to use for encrypting the secret content stored inside the vault.
- OCID of the compartment where the vault resides. This defaults to the compartment of the notebook session when
  used in a Data Science notebook session.


**BDSSecretKeeper**

You can also save the connection parameters as well as the files needed to configure the kerberos
authentication into vault. This will allow you to use repetitively in
different notebook sessions, machines, and Jobs.

``BDSSecretKeeper`` requires the following fields: 

- ``principal``: str. The unique identity to which Kerberos can assign tickets. 
- ``hdfs_host``: str. The hdfs host name from the bds cluster.
- ``hive_host``: str. The hive host name from the bds cluster. 
- ``hdfs_port``: str. The hdfs port from the bds cluster. 
- ``hive_port``: str. The hive port from the bds cluster. 
- ``kerb5_path``: str. The ``krb5.conf`` file path.
- ``keytab_path``: str. The path to the keytab file.
- ``vault_id: str``. OCID of the vault.
- ``key_id: str``. OCID of the master key used for encrypting the secret.
- ``compartment_id: str``. OCID of the compartment where the vault is located. This defaults to the compartment of the notebook session when
  used in a Data Science notebook session.


**BDSSecretKeeper.save**

``BDSSecretKeeper.save`` API serializes and stores the credentials to Vault using the following parameters:

- ``name (str)`` – Name of the secret when saved in Vault.
- ``description (str)`` – Description of the secret when saved in Vault.
- ``freeform_tags (dict, optional)``. Default None. Free form tags to use for saving the secret in the OCI Console.
- ``defined_tags (dict, optional.)``. Default None. Save the tags under predefined tags in the OCI Console.
- ``save_files (bool, optional.)``. Default True. If set to True, then the keytab and kerb5 config files are serialized and saved.

Examples
++++++++

**Saving a Secret With the Keytab and kerb5 config Files**

.. code:: ipython3

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
    vault_id = "ocid1.vault.oc1.iad.*********"
    key_id = "ocid1.key.oc1.iad.*********"

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



**Saving a Secret Without Saving the Keytab and kerb5 config File**

.. code:: ipython3

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
    vault_id = "ocid1.vault.oc1.iad.*********"
    key_id = "ocid1.key.oc1.iad.*********"

    bds_keeper = BDSSecretKeeper(
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

    saved_secret = bds_keeper.save(name="your_bds_config_secret_name",
                            description="your bds credentials",
                            freeform_tags={"schema":"emp"},
                            defined_tags={},
                            save_files=False)

    print(saved_secret.secret_id)

``'ocid1.vaultsecret.oc1..<unique_ID>'``

Loading Credentials
-------------------

**Prerequisite**

- OCID of the secret stored in vault.

**BDSSecretKeeper.load_secret**

``BDSSecretKeeper.load_secret`` API deserializes and loads the credentials from Vault. You could use this API in one of
the following ways -

Using a ``with`` statement:

.. code:: python3

    with BDSSecretKeeper.load_secret('ocid1.vaultsecret.oc1..<unique_ID>') as bdssecret:
        print(bdssecret['hdfs_host'])

Without using a ``with`` statement:

.. code:: python3

    bdssecretobj = BDSSecretKeeper.load_secret('ocid1.vaultsecret.oc1..<unique_ID>')
    bdssecret = bdssecretobj.to_dict()
    print(bdssecret['hdfs_host'])


``load_secret`` takes following parameters -

- ``source``: Either the file that was exported from ``export_vault_details`` or the OCID of the secret
- ``format``: Optional. If ``source`` is a file, then this value must be ``json`` or ``yaml`` depending on the file format.
- ``export_env``: Default is False. If set to True, the credentials are exported as environment variable when used with
  the ``with`` operator.
- ``export_prefix``: The default name for environment variable is user_name, password, service_name, and wallet_location. You
  can add a prefix to avoid name collision
- ``auth``: Provide overriding authorization information if the authorization information is different from the ``ads.set_auth`` setting.
- ``keytab_dir``: Optional. Directory path where the ``keytab`` ZIP file is saved after the contents are retrieved from the vault. If the ``keytab`` content is not available in the specified secret OCID, then this attribute is ignored.

If the ``keytab`` and kerb5 configuration files were saved in the vault, then a ``keytab`` and kerb5 configuration file of the same name is created by ``load_secret``. By default, the ``keytab`` file is created in the ``keytab_path`` specified in the secret.
To update the location, set the directory path with ``key_dir``. However, the kerb5 configuration file is always saved in the "~/.bds_config/krb5.conf" path.

Note that ``keytab`` and kerb5 configuration files are saved only when the
content is saved into the vault.

After you load and save the configuration parameters files, you can
call the ``krbcontext`` context manager to create a Kerberos ticket.


Examples
++++++++

**Access Credentials Using a With Statement**

To specify a local ``keytab`` file, set the path to the ZIP file with ``wallet_location``:

.. code:: ipython3

    from pyhive import hive
    
    with BDSSecretKeeper.load_secret(saved_secret.secret_id, keytab_dir="~/path/to/save/keytab_file/") as cred:
        with krbcontext(principal=cred["principal"], keytab_path=cred['keytab_path']):
            hive_cursor = hive.connect(host=cred["hive_host"],
                                       port=cred["hive_port"],
                                       auth='KERBEROS',
                                       kerberos_service_name="hive").cursor()



Now you can query the data from Hive:

.. code:: ipython3

    hive_cursor.execute("""
        select *
        from your_db.your_table
        limit 10
    """)
    
    import pandas as pd
    pd.DataFrame(hive_cursor.fetchall(), columns=[col[0] for col in hive_cursor.description])

**Access Credentials Without Using a With Statement**

Loading from secret id:

.. code:: ipython3

    bdssecretobj = BDSSecretKeeper.load_secret(saved_secret.secret_id)
    bdssecret = bdssecretobj.to_dict()
    print(bdssecret)

Loading from a JSON file:

.. code:: ipython3

    bdssecretobj = BDSSecretKeeper.load_secret(source="./my_bds_vault_info.json", format="json")
    bdssecretobj.to_dict()

Loading from a YAML file:

.. code:: ipython3

    bdssecretobj = BDSSecretKeeper.load_secret(source="./my_bds_vault_info.yaml", format="yaml")
    bdssecretobj.to_dict()