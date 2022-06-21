.. code-block:: python3

    import ads
    import fsspec
    import os

    from ads.bds.auth import refresh_ticket

    ads.set_auth('resource_principal')
    refresh_ticket(principal="<your_principal>", keytab_path="<your_local_keytab_file_path>", 
                   kerb5_path="<your_local_kerb5_config_file_path>")
    cursor = hive.connect(host="<hive_host>", port="<hive_port>",
                          auth='KERBEROS', kerberos_service_name="hive").cursor()





