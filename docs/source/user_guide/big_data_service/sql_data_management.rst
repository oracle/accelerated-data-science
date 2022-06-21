SQL Data Management
*******************

This section demonstrates how to perform standard SQL-based data management operations in BDS using various frameworks, see the individual framework's documentation for details.

A Kerberos ticket is needed to :ref:`connect to the BDS cluster<BDS Connect>`. You can obtain this authentication ticket with the ``refresh_ticket()`` method, or with the use of the vault and a ``BDSSercretKeeper`` object. This section demonstrates the use of the ``BDSSecretKeeper`` object because this is more secure and is the recommended method.

Ibis
====

`Ibis <https://github.com/ibis-project/ibis>`_ is an open-source library by `Cloudera <https://www.cloudera.com/>`_ that provides a Python framework to access data and perform analytical computations from different sources. The `Ibis project <https://ibis-project.org/docs/dev/>`_ is designed to provide an abstraction over different dialects of SQL. It enables the data scientist to interact with many different data systems. Some of these systems are Dask, MySQL, Pandas, PostgreSQL, PySpark, and most importantly for use with BDS, Hadoop clusters. 

Connect
-------

Obtaining a Kerberos ticket, depending on your system configuration, you may need to define the ``ibis.options.impala.temp_db`` and ``ibis.options.impala.temp_hdfs_path`` options. The ``ibis.impala.connect()`` method makes a connection to the `Impala execution backend <https://ibis-project.org/docs/dev/backends/Impala/>`_. The ``.sql()`` allows you to run SQL commands on the data.

.. code-block:: python3

    import ibis
    
    with BDSSecretKeeper.load_secret("<secret_id>") as cred:
        with krbcontext(principal=cred["principal"], keytab_path=cred['keytab_path']):
            ibis.options.impala.temp_db = '<temp_db>'
            ibis.options.impala.temp_hdfs_path = '<temp_hdfs_path>'
            hdfs = ibis.impala.hdfs_connect(host=cred['hdfs_host'], port=cred['hdfs_port'],
                                             use_https=False, verify=False,
                                             auth_mechanism='GSSAPI', protocol='webhdfs')
            client = ibis.impala.connect(host=cred['hive_host'], port=cred['hive_port'],
                                         hdfs_client=hdfs, auth_mechanism="GSSAPI",
                                         use_ssl=False, kerberos_service_name="hive")


Query
-----

To query the data using ibis use an SQL DML command like ``SELECT``. Pass the string to the ``.sql()`` method, and then call ``.execute()`` on the returned object. The output is a Pandas dataframe.

.. code-block:: python3

    df = client.sql("SELECT * FROM bikes.trips LIMIT 100").execute(limit=None)


Close a Connection
------------------

It is important to close sessions when you don't need them anymore. This frees up resources in the system. Use the ``.close()`` method close sessions.

.. code-block:: python3

    client.close()

Impala
======

`Impala <https://github.com/cloudera/impyla>`_ is a Python client for `HiveServer2 <https://cwiki.apache.org/confluence/display/hive/hiveserver2+overview>`_ implementations (i.e. Impala, Hive). Both Impala and PyHive clients are HiveServer2 compliant so the connection syntax is very similar. The difference is that the Impala client uses the Impala query engine and PyHive uses Hive. In practical terms, Hive is best suited for long-running batch queries and Impala is better suited for real-time interactive querying, see  `more about the differences between Hive and Impala <https://www.topcoder.com/thrive/articles/the-relationship-between-impala-and-hive-and-its-application-in-business>`_.

The Impala ``dbapi`` module is a `Python DB-API <http://www.python.org/dev/peps/pep-0249/>`_ interface.

Connect
-------

After obtaining a Kerberos ticket, use the ``connect()`` method to make the connection. It returns a connection, and the ``.cursor()`` method returns a cursor object. The cursor has the method ``.execute()`` that allows you to run Impala SQL commands on the data.

.. code-block:: python3

    from impala.dbapi import connect
    
    with BDSSecretKeeper.load_secret("<secret_id>") as cred:
        with krbcontext(principal=cred["principal"], keytab_path=cred['keytab_path']):
            cursor = connect(host=cred["hive_host"], port=cred["hive_port"], 
                             auth_mechanism="GSSAPI", kerberos_service_name="hive").cursor()

Create a Table
--------------

To create an Impala table and insert data, use the ``.execute()`` method on the cursor object, and pass in Impala SQL commands to perform these operations.

.. code-block:: python3

    cursor.execute("CREATE TABLE default.location (city STRING, province STRING)")
    cursor.execute("INSERT INTO default.location VALUES ('Halifax', 'Nova Scotia')")

Query
-----

To query an Impala table, use an Impala SQL DML command like ``SELECT``. Pass this string to the ``.execute()`` method on the cursor object to create a record set in the cursor. You can obtain a Pandas dataframe with the ``as_pandas()`` function.

.. code-block:: python3

    from impala.util import as_pandas

    cursor.execute("SELECT * FROM default.location")
    df = as_pandas(cursor)


Drop a Table
------------

To drop an Impala table, use an Impala SQL DDL command like ``DROP TABLE``. Pass this string to the ``.execute()`` method on the cursor object.

.. code-block:: python3

    cursor.execute("DROP TABLE IF EXISTS default.location")

Close a Connection
------------------

It is important to close sessions when you don't need them anymore. This frees up resources in the system. Use the ``.close()`` method on the cursor object to close a connection.

.. code-block:: python3

    cursor.close()

PyHive
======

`PyHive <https://github.com/dropbox/PyHive>`_ is a set of interfaces to Presto and Hive. It is based on the `SQLAlchemy <http://www.sqlalchemy.org/>`_ and `Python DB-API <http://www.python.org/dev/peps/pep-0249/>`_ interfaces for `Presto <https://prestodb.io/>`_ and `Hive <http://hive.apache.org/>`_.

Connect
-------

After obtaining a Kerberos ticket, call the ``hive.connect()`` method to make the connection. It returns a connection, and the ``.cursor()`` method returns a cursor object. The cursor has the ``.execute()`` method that allows you to run Hive SQL commands on the data.

.. include:: _template/hive_connect_with_vault.rst

Create a Table
--------------

To create a Hive table and insert data, use the ``.execute()`` method on the cursor object and pass in Hive SQL commands to perform these operations.

.. code-block:: python3

    cursor.execute("CREATE TABLE default.location (city STRING, province STRING)")
    cursor.execute("INSERT INTO default.location VALUES ('Halifax', 'Nova Scotia')")

Query
-----

To query a Hive table, use a Hive SQL DML command like ``SELECT``. Pass this string to the ``.execute()`` method on the cursor object. This creates a record set in the cursor. You can access the actual records with methods like ``.fetchall()``, ``.fetchmany()``, and ``.fetchone()``.

In the following example, the ``.fetchall()`` method is used in a ``pd.DataFrame()`` call to return all the records in Pandas dataframe:
.

.. code-block:: python3

    import pandas as pd

    cursor.execute("SELECT * FROM default.location")
    df = pd.DataFrame(cursor.fetchall(), columns=[col[0] for col in cursor.description])


Drop a Table
------------

To drop a Hive table, use a Hive SQL DDL command like ``DROP TABLE``. Pass this string to the ``.execute()`` method on the cursor object.

.. code-block:: python3

    cursor.execute("DROP TABLE IF EXISTS default.location")

Close a Connection
------------------

It is important to close sessions when you don't need them anymore. This frees up resources in the system. Use the ``.close()`` method on the cursor object to close a connection.

.. code-block:: python3

    cursor.close()


