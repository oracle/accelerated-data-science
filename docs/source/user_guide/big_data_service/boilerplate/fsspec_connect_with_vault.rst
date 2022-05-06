.. code:: ipython3

    import ads
    import fsspec

    from ads.secrets.big_data_service import BDSSecretKeeper
    from ads.bds.auth import has_kerberos_ticket, krbcontext
    
    ads.set_auth("resource_principal")
    with BDSSecretKeeper.load_secret("<secret_id>") as cred:
        with krbcontext(principal = cred["principal"], keytab_path = cred['keytab_path']):
            hdfs_config = {
                "protocol": "webhdfs",
                "host": cred["hdfs_host"],
                "port": cred["hdfs_port"],
                "kerberos": "True"
            }            

    fs = fsspec.filesystem(**hdfs_config)

