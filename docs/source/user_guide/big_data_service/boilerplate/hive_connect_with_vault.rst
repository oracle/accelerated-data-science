.. code:: python3

    import ads
    import os
    
    from ads.bds.auth import krbcontext
    from ads.secrets.big_data_service import BDSSecretKeeper
    from pyhive import hive
    
    ads.set_auth('resource_principal')
    with BDSSecretKeeper.load_secret("<secret_id>") as cred:
        with krbcontext(principal=cred["principal"], keytab_path=cred['keytab_path']):
            cursor = hive.connect(host=cred["hive_host"],
                                  port=cred["hive_port"],
                                  auth='KERBEROS',
                                  kerberos_service_name="hive").cursor()

