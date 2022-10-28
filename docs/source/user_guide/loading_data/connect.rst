#########
Load Data
#########


Connecting to Data Sources
**************************

You can load data into ADS in several different ways from Oracle Cloud Infrastructure Object Storage, cx_Oracle, or S3.  Following are some examples.

Begin by loading the required libraries and modules:

.. code-block:: python3

    import ads
    import numpy as np
    import pandas as pd
    from ads.common.auth import default_signer

Object Storage
==============

To load a dataframe from Object Storage using the API keys, you can use the following example, replacing the angle bracketed content with the location and name of your file:

.. code-block:: python3

  ads.set_auth(auth="api_key", oci_config_location="~/.oci/config", profile="DEFAULT")
  bucket_name = <bucket-name>
  file_name = <file-name>
  namespace = <namespace>
  df = pd.read_csv(f"oci://{bucket_name}@{namespace}/{file_name}", storage_options=default_signer())

For a list of ``pandas`` functions to read different file format, please refer to `the Pandas documentation <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>`_.

To load a dataframe from Object Storage using the resource principal method, you can use the following example, replacing the angle bracketed content with the location and name of your file:

.. code-block:: python3

  ads.set_auth(auth='resource_principal')
  bucket_name = <bucket-name>
  file_name = <file-name>
  namespace = <namespace>
  df = pd.read_csv(f"oci://{bucket_name}@{namespace}/{file_name}", storage_options=default_signer())

To write a pandas dataframe to object storage, provide the file name in the following format -  ``oci://<mybucket>@<mynamespace>/<path/to/flle/name>``

.. code-block:: python3

  ads.set_auth(auth='resource_principal')
  bucket_name = <bucket-name>
  file_name = <file-name>
  namespace = <namespace>
  df = pd.to_csv(f"oci://{bucket_name}@{namespace}/{file_name}", index=False, storage_options=default_signer())

  # To setup the content type while writing to object storage, set ``oci_additional_kwargs`` attribute with ``storage_options`` to the desired content type

  storage_optons = default_signer()
  storage_options['oci_additional_kwargs'] = {"content_type":"application/octet-stream"}
  df = pd.to_csv(f"oci://{bucket_name}@{namespace}/{file_name}", index=False, storage_options=storage_options)


Local Storage
=============

To load a dataframe from a local source, use functions from ``pandas`` directly:

.. code-block:: python3

  df = pd.read_csv("/path/to/data.data")

Oracle Database
===============

.. image:: images/cx_Oracle.jpeg
  :height: 150
  :alt: cx_Oracle Logo

When using the `Oracle ADB <https://www.oracle.com/database/>`_ with Python the most common representation of tabular data is  a `Pandas dataframe <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_. When you're in a dataframe, you can perform many operations from visualization to persisting in a variety of formats.

Oracle ADB to Pandas
--------------------

The Pandas ``read_sql(...)`` function is a general, database independent approach that uses the `SQLAlchemy - Object Relational Mapper <https://www.sqlalchemy.org/>`_ to arbitrate between specific database types and Pandas.

..

   Read SQL query or database table into a dataframe.

   This function is a convenience wrapper around read_sql_table and ``read_sql_query`` (for backward compatibility). It delegates to the specific function depending on the provided input. A SQL query is routed to read_sql_query, while a database table name is routed to ``read_sql_table``.

Use the Pandas ADS accessor drop-in replacement, ``pd.DataFrame.ads.read_sql(...)``, instead of using ``pd.read_sql``.

See :doc:`how to <../secrets/autonomous_database>` save and retrieve credentials from OCI Vault

**Example using Wallet File**

  .. code-block:: python3 

        # If you are using Wallet file, provide the zip file path for `wallet_location`
        connection_parameters = {
            "user_name": "<username>",
            "password": "<password>",
            "service_name": "<service_name_{high|med|low}>",
            "wallet_location": "/full/path/to/my_wallet.zip",
        }
        import pandas as pd
        import ads

        # simple read of a SQL query into a dataframe with no bind variables    
        df = pd.DataFrame.ads.read_sql(
            "SELECT * FROM SH.SALES",
            connection_parameters=connection_parameters,
        )

        # read of a SQL query into a dataframe with a bind variable. Use bind variables
        # rather than string substitution to avoid the SQL injection attack vector.
        df = pd.DataFrame.ads.read_sql(
            """
            SELECT
            *
            FROM
            SH.SALES
            WHERE
                ROWNUM <= :max_rows
            """,
            bind_variables={
                "max_rows" : 100
            }
            ,
            connection_parameters=connection_parameters,
        )


**Example using TLS**

  .. code-block:: python3 

        connection_parameters = {
            "user_name": "<username>",
            "password": "<password>",
            "dsn": "<connection string copied from console>",
        }
        import pandas as pd
        import ads

        # simple read of a SQL query into a dataframe with no bind variables    
        df = pd.DataFrame.ads.read_sql(
            "SELECT * FROM SH.SALES",
            connection_parameters=connection_parameters,
        )

        # read of a SQL query into a dataframe with a bind variable. Use bind variables
        # rather than string substitution to avoid the SQL injection attack vector.
        df = pd.DataFrame.ads.read_sql(
            """
            SELECT
            *
            FROM
            SH.SALES
            WHERE
                ROWNUM <= :max_rows
            """,
            bind_variables={
                "max_rows" : 100
            }
            ,
            connection_parameters=connection_parameters,
        )


Oracle Database to Pandas - No Wallet
-------------------------------------

.. versionadded:: 2.5.6.

If your database connection doesn't require a wallet file, you can connect to the database by specifying ``host/port/sid/service name``.

See :doc:`how to <../secrets/oracle>` save and retrieve credentials from OCI Vault


**Example**

.. code-block:: python3 

        connection_parameters = {
            "user_name": "<username>",
            "password": "<password>",
            "service_name": "<service_name>",
            "host": "<database hostname>",
            "port": "<database port number>""
        }
        import pandas as pd
        import ads

        # simple read of a SQL query into a dataframe with no bind variables    
        df = pd.DataFrame.ads.read_sql(
            "SELECT * FROM SH.SALES",
            connection_parameters=connection_parameters,
        )

        # read of a SQL query into a dataframe with a bind variable. Use bind variables
        # rather than string substitution to avoid the SQL injection attack vector.
        df = pd.DataFrame.ads.read_sql(
            """
            SELECT
            *
            FROM
            SH.SALES
            WHERE
                ROWNUM <= :max_rows
            """,
            bind_variables={
                max_rows : 100
            }
            ,
            connection_parameters=connection_parameters,
        )

Performance
-----------

The performance is limited by three things:

* Generational latency: How long the database takes to return rows, use of indexes and writing efficient SQL mitigates this performance bottleneck.
* Network saturation: Once the network is saturated, data can't be delivered between the database and notebook environment any faster. OCI networking is very fast and this isn't usually a concern. One exception is when the network path goes over VPN or other more complex routing topologies.
* CPU latency in the notebook: Python has to collect the byte stream delivered by the database into Python data types before being promoted to Numpy objects for Pandas. Additionally, there is a cryptographic CPU overhead because the data in transit is secured with public key infrastructure (PKI).

Large Result Set
----------------

If a database query returns more rows than the memory of the client permits, you have a couple of options. The simplest is to use a larger client shape, along with increased compute performance because larger shapes come with more RAM. If that's not an option, then you can use the ``pd.DataFrame.ads.read_sql`` mixin in chunk mode, where the result is no longer a Pandas dataframe it is an iterator over a sequence of dataframes. You could use this read a large data set and write it to Object storage or a local file system with the following example:

  .. code-block:: python3 

       for i, df in enumerate(pd.DataFrame.ads.read_sql(
               "SELECT * FROM SH.SALES",
               chunksize=100000 # rows per chunk,
               connection_parameters=connection_parameters,
             ))
          # each df will contain up to 100000 rows (chunksize)
          # to write the data to object storage use oci://bucket@namespace/part_{i}.csv"
          df.to_csv(f"part_{i}.csv")

Very Large Result Set
---------------------

If the data exceeds what's practical in a notebook, then the next step is to use the `Data Flow service <https://www.oracle.com/big-data/data-flow/>`_ to partition the data across multiple nodes and handle data of any size up to the size of the cluster.
   
Pandas to Oracle Database
-------------------------

Typically, you would do this using ``df.to_sql``\. However, this uses Oracle Resource Manager to collect data and is less efficient than code that has been optimized for a specific database.

Instead, use the Pandas ADS accessor mixin.

With a ``df`` dataframe, writing this to the database is as simple as:

.. code-block:: python3

       df.ads.to_sql(
           "MY_TABLE",
           connection_parameters=connection_parameters, # Should contain wallet location if you are connecting to ADB
           if_exists="replace"
       )

The resulting data types (if the table was created by ADS as opposed to inserting into an existing table), are governed by the following:

.. list-table::
   :header-rows: 1

   * - Pandas
     - Oracle
   * - bool
     - NUMBER(1)
   * - int16
     - INTEGER
   * - int32
     - INTEGER
   * - int64
     - INTEGER
   * - float16
     - FLOAT
   * - float32
     - FLOAT
   * - float64
     - FLOAT
   * - datetime64
     - TIMESTAMP
   * - string
     - VARCHAR2 (Maximum length of the actual data.)


When a table is created, the length of any ``VARCHAR2`` column is computed from the longest string in the column. The ORM defaults to ``CLOB`` data, which is not correct or efficient. CLOBS are stored efficiently by the database, but the c API to query them works differently. The non-LOB columns are returned to the client through a cursor, but LOBs are handled differently resulting in an additional network fetch per row, per LOB column. ADS deals with this by creating the correct data type, and setting the correct ``VARCHAR2`` length.

MySQL
=====

.. versionadded:: 2.5.6.

To load a dataframe from a MySQL database, you must set ``engine=mysql`` in ``pd.DataFrame.ads.read_sql``.

See :doc:`how to <../secrets/mysql>` save and retrieve credentials from OCI Vault


**Example**

  .. code-block:: python3 

        connection_parameters = {
            "user_name": "<username>",
            "password": "<password>",
            "host": "<database hostname>",
            "port": "<database port number>",
            "database": "<database name>"
        }
        import pandas as pd
        import ads

        # simple read of a SQL query into a dataframe with no bind variables    
        df = pd.DataFrame.ads.read_sql(
            "SELECT * FROM EMPLOYEE",
            connection_parameters=connection_parameters,
            engine="mysql"
        )

        # read of a SQL query into a dataframe with a bind variable. Use bind variables
        # rather than string substitution to avoid the SQL injection attack vector.
        df = pd.DataFrame.ads.read_sql(
            """
            SELECT
            *
            FROM
            EMPLOYEE
            WHERE
                emp_no <= ?
            """,
            bind_variables=(1000,)
            ,
            connection_parameters=connection_parameters,
            engine="mysql"
        )

To save the dataframe ``df`` to MySQL, use ``df.ads.to_sql`` API with ``engine=mysql``

.. code-block:: python3

       df.ads.to_sql(
           "MY_TABLE",
           connection_parameters=connection_parameters,
           if_exists="replace",
           engine="mysql"
       )

The resulting data types (if the table was created by ADS as opposed to inserting into an existing table), are governed by the following:

.. list-table::
   :header-rows: 1

   * - Pandas
     - MySQL
   * - bool
     - NUMBER(1)
   * - int16
     - INTEGER
   * - int32
     - INTEGER
   * - int64
     - INTEGER
   * - float16
     - FLOAT
   * - float32
     - FLOAT
   * - float64
     - FLOAT
   * - datetime64
     - DATETIME (Format: *%Y-%m-%d %H:%M:%S*)
   * - string
     - VARCHAR (Maximum length of the actual data.)

BDS Hive
========

.. versionadded:: 2.6.1.

To load a dataframe from BDS Hive, set ``engine="hive"`` in ``pd.DataFrame.ads.read_sql``.

See :doc:`how to <../secrets/big_data_service>` save and retrieve credentials from OCI Vault


Connection Parameters
---------------------

**Work with BDS with Kerberos authentication**

If you are working with BDS that requires Kerberos authentication, you can follow :ref:`here <BDS Connect>` to get connection parameters required to connect with BDS, and then follow :ref:`here <secretbds>` to save the connection parameters as well as the files needed to configure the 
kerberos authentication into vault. The ``connection_parameters`` can be set as:

.. code-block:: python3

    connection_parameters = {
        "host": "<hive hostname>",
        "port": "<hive port number>",
    }

**Work with unsecure BDS**

If you are working with unsecure BDS, you can set ``connection_parameters`` as:

.. code-block:: python3

    connection_parameters = {
        "host": "<hive hostname>",
        "port": "<hive port number>",
        "auth_mechanism": "PLAIN" # for connection with unsecure BDS
    }

**Example**

.. code-block:: python3

    connection_parameters = {
        "host": "<database hostname>",
        "port": "<database port number>",
    }
    import pandas as pd
    import ads
    
    # simple read of a SQL query into a dataframe with no bind variables
    df = pd.DataFrame.ads.read_sql(
        "SELECT * FROM EMPLOYEE",
        connection_parameters=connection_parameters,
        engine="hive"
    )
    
    # read of a SQL query into a dataframe with a bind variable. Use bind variables
    # rather than string substitution to avoid the SQL injection attack vector.
    df = pd.DataFrame.ads.read_sql(
        """
        SELECT
        *
        FROM
        EMPLOYEE
        WHERE
            `emp_no` <= ?
        """,
        bind_variables=(1000,)
        ,
        connection_parameters=connection_parameters,
        engine="hive"
    )

To save the dataframe ``df`` to BDS Hive, use ``df.ads.to_sql`` API with ``engine="hive"``.

.. code-block:: python3

    df.ads.to_sql(
        "MY_TABLE",
        connection_parameters=connection_parameters,
        if_exists="replace",
        engine="hive"
    )

Partition
---------

You can create table with partition, and then use ``df.ads.to_sql`` API with ``engine="hive"``, ``if_exists="append"`` to insert data into the table.

.. code-block:: python3

    create_table_sql = f'''
                        CREATE TABLE {table_name} (col1_name datatype, ...)
                        partitioned by (col_name datatype, ...)
                        '''
    
    df.ads.to_sql(
        "MY_TABLE",
        connection_parameters=connection_parameters,
        if_exists="append",
        engine="hive"
    )

Large Dataframe
---------------

If the dataframe waiting to be uploaded has many rows, and the ``.to_sql()`` method is slow, you have other options. The simplest is
to use a larger client shape, along with increased compute performance because larger shapes come with more RAM. If that’s not an option, then you can follow these steps:

.. code-block:: python3

    # Step1: Save your df as csv
    df.to_csv(f"my_data.csv")
    
    # Step2: Upload the csv to hdfs
    hdfs_host = "<hdfs hostname>"
    hdfs_port = "<hdfs port number>"
    hdfs_config = {"host": hdfs_host, "port": hdfs_port, "protocol": "webhdfs"}
    fs = fsspec.filesystem(**hdfs_config)
    fs.upload(
        lpath="./my_data.csv",
        rpath="/user/hive/iris.csv"
    )
    
    # Step3: Create table
    sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (col1_name datatype, ...)
    ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
    STORED AS TEXTFILE
    """
    cursor.execute(sql)
    
    # Step4: Load data into Hive table from hdfs
    hdfs_path = "./my_data.csv"
    sql = f"LOAD DATA INPATH '{hdfs_path}' INTO TABLE {table_name}"
    cursor.execute(sql)

HTTP(S) Sources
===============

To load a dataframe from a remote web server source, use ``pandas`` directly and specify the URL of the data:

.. code-block:: python3

  df = pd.read_csv('https://example.com/path/to/data.csv')

Convert Pandas DataFrame to ``ADSDataset``
==========================================

To convert a Pandas dataframe to ``ADSDataset``, pass the ``pandas.DataFrame`` object directly into the ADS ``DatasetFactory.open`` method:

.. code-block:: python3

  import pandas as pd
  from ads.dataset.factory import DatasetFactory

  df = pd.read_csv('/path/some_data.csv) # load data with Pandas

  # use open...

  ds = DatasetFactory.open(df) # construct **ADS** Dataset from DataFrame

  # alternative form...

  ds = DatasetFactory.from_dataframe(df)

  # an example using Pandas to parse data on the clipboard as a CSV and construct an ADS Dataset object
  # this allows easily transfering data from an application like Microsoft Excel, Apple Numbers, etc.

  ds = DatasetFactory.from_dataframe(pd.read_clipboard())

  # use Pandas to query a SQL database:

  from sqlalchemy import create_engine
  engine = create_engine('dialect://user:pass@host:port/schema', echo=False)
  df = pd.read_sql_query('SELECT * FROM mytable', engine, index_col = 'ID')
  ds = DatasetFactory.from_dataframe(df)


Using ``PyArrow``
=================

ADS supports reading files into ``PyArrow`` dataset directly via ``ocifs``. ``ocifs`` is installed as ADS dependencies.

.. code-block:: python3

  import ocifs
  import pyarrow.dataset as ds
  bucket_name = <bucket_name>
  namespace = <namespace>
  path = <path>
  fs = ocifs.OCIFileSystem(**default_signer())
  ds = ds.dataset(f"{bucket_name}@{namespace}/{path}/", filesystem=fs)

DataSetFactory
**************

.. toctree::
   :maxdepth: 2

   connect_legacy
