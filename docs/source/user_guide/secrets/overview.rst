Overview
********

Services such as OCI Database and Streaming require users to provide credentials. These credentials must be safely accessed at runtime. `OCI Vault <https://docs.oracle.com/en-us/iaas/Content/KeyManagement/Concepts/keyoverview.htm>`_ provides a mechanism for safe storage and access of secrets. ``SecretKeeper`` uses Vault as a backend to store and retrieve the credentials. The data structure of the credentials varies from service to service. There is a ``SecretKeeper`` specific to each data structure.

These classes are provided:

* ``ADBSecretKeeper``: Stores credentials for the Oracle Autonomous Database, with or without the wallet file.
* ``AuthTokenSecretKeeper``: Stores an Auth Token or Access Token string. This could be an Auth Token to use to connect to Streaming, Github, or other systems that used Auth Tokens or Access Token strings.
* ``BDSSecretKeeper``: Stores credentials for Oracle Big Data Service with or without Keytab and kerb5 configuration files.
* ``MySQLDBSecretKeeper``: Stores credentials for the MySQL database. This class will work with many databases that authenticate with a username and password only.
* ``OracleDBSecretKeeper``: Stores credentials for the Oracle Database. 

