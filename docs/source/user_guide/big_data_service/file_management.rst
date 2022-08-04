File Management
***************

This section demonstrates various methods to work with files on BDS' HDFS, see the individual framework's documentation for details.

A Kerberos ticket is needed to :ref:`connect to the BDS cluster<BDS Connect>`. This authentication ticket can be obtained with the ``refresh_ticket()`` method or with the use of the Vault and a ``BDSSercretKeeper`` object. This section will demonstrate the use of the ``BDSSecretKeeper`` object as this is more secure and is the preferred method.

FSSpec
======

The ``fsspec`` or `Filesystem Spec <https://filesystem-spec.readthedocs.io/en/latest/>`_ is an interface that allows access to local, remote, and embedded file systems. You use it to access data stored in the BDS' HDFS. This connection is made with the `WebHDFS <https://hadoop.apache.org/docs/r1.0.4/webhdfs.html>`_ protocol. 

The ``fsspec`` library must be able to access BDS so a Kerberos ticket must be generated. The secure and recommended method to do this is to use ``BDSSecretKeeper`` that stores the BDS credentials in the vault not the notebook session.

This section outlines some common file operations, see the ``fsspec`` `API Reference <https://filesystem-spec.readthedocs.io/en/latest/api.html>`_ for complete details on the features that are demonstrated and additional functionality. 

:ref:`Pandas <BDS Pandas>` and :ref:`PyArrow <BDS PyArrow>` can also use ``fsspec`` to perform file operations. 

Connect
-------

Credentials and configuration information is stored in the vault. This information is used to obtain a Kerberos ticket and define the ``hdfs_config`` dictionary. This configuration dictionary is passed to the `fsspec.filesystem() <https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.filesystem>`_ method to make a connection to the BDS' underlying HDFS storage.

.. include:: _template/fsspec_connect_with_vault.rst   


Delete
------

Delete files from HDFS using the `.rm() <https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.rm>`_ method. It accepts a path of the files to delete.

.. code-block:: python3

    fs.rm("/data/biketrips/2020??-tripdata.csv", recursive=True)

Download
--------

Download files from HDFS to a local storage device using the `.get() <https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.get>`_ method. It takes the HDFS path of the files to download, and the local path to store the files.

.. code-block:: python3

    fs.get("/data/biketrips/20190[123456]-tripdata.csv", local_path="./first_half/", overwrite=True)

List
----

The `.ls() <https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.archive.AbstractArchiveFileSystem.ls>`_ method lists files. It returns the matching file names as a list.

.. code-block:: python3

    fs.ls("/data/biketrips/2019??-tripdata.csv")


.. parsed-literal::

    ['201901-tripdata.csv',
     '201902-tripdata.csv',
     '201903-tripdata.csv',
     '201904-tripdata.csv',
     '201905-tripdata.csv',
     '201906-tripdata.csv',
     '201907-tripdata.csv',
     '201908-tripdata.csv',
     '201909-tripdata.csv',
     '201910-tripdata.csv',
     '201911-tripdata.csv',
     '201912-tripdata.csv']

Upload
------

The `.put() <https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.put>`_ method is used to upload files from local storage to HDFS. The first parameter is the local path of the files to upload. The second parameter is the HDFS path where the files are to be stored. 
`.upload() <https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.upload>`_ is an alias of `.put()`.
.. code-block:: python3

    fs.put(
        lpath="./first_half/20200[456]-tripdata.csv",
        rpath="/data/biketrips/second_quarter/"
)

Ibis
====

`Ibis <https://github.com/ibis-project/ibis>`_ is an open-source library by `Cloudera <https://www.cloudera.com/>`_ that provides a Python framework to access data and perform analytical computations from different sources. Ibis allows access to the data ising HDFS. You use the ``ibis.impala.hdfs_connect()`` method to make a connection to HDFS, and it returns a handler. This handler has methods such as ``.ls()`` to list, ``.get()`` to download, ``.put()`` to upload, and ``.rm()`` to delete files. These operations support globbing. Ibis' HDFS connector supports a variety of `additional operations <https://ibis-project.org/docs/dev/backends/Impala/#hdfs-interaction>`_.

Connect
-------

After obtaining a Kerberos ticket, the ``hdfs_connect()`` method allows access to the HDFS. It is a thin wrapper around a `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_ file system. Depending on your system configuration, you may need to define the ``ibis.options.impala.temp_db`` and ``ibis.options.impala.temp_hdfs_path`` options. 

.. code-block:: python3

    import ibis
    
    with BDSSecretKeeper.load_secret("<secret_id>") as cred:
        with krbcontext(principal=cred["principal"], keytab_path=cred['keytab_path']):
            hdfs = ibis.impala.hdfs_connect(host=cred['hdfs_host'], port=cred['hdfs_port'],
                                                 use_https=False, verify=False, 
                                                 auth_mechanism='GSSAPI', protocol='webhdfs')

Delete
------

Delete files from HDFS using the `.rm() <https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.rm>`_ method. It accepts a path of the files to delete.

.. code-block:: python3

    hdfs.rm("/data/biketrips/2020??-tripdata.csv", recursive=True)

Download
--------

Download files from HDFS to a local storage device using the `.get() <https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.get>`_ method. It takes the HDFS path of the files to download, and the local path to store the files.

.. code-block:: python3

    hdfs.get("/data/biketrips/20190[123456]-tripdata.csv", local_path="./first_half/", overwrite=True)

List 
----

The `.ls() <https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.archive.AbstractArchiveFileSystem.ls>`_ method lists files. It returns the matching file names as a list.

.. code-block:: python3

    hdfs.ls("/data/biketrips/2019??-tripdata.csv")


.. parsed-literal::

    ['201901-tripdata.csv',
     '201902-tripdata.csv',
     '201903-tripdata.csv',
     '201904-tripdata.csv',
     '201905-tripdata.csv',
     '201906-tripdata.csv',
     '201907-tripdata.csv',
     '201908-tripdata.csv',
     '201909-tripdata.csv',
     '201910-tripdata.csv',
     '201911-tripdata.csv',
     '201912-tripdata.csv']


Upload
------

Use the `.put() <https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.put>`_ method to upload files from local storage to HDFS. The first parameter is the HDFS path where the files are to be stored. The second parameter is the local path of the files to upload.

.. code-block:: python3

    hdfs.put(rpath="/data/biketrips/second_quarter/", 
             lpath="./first_half/20200[456]-tripdata.csv", 
             overwrite=True, recursive=True)

.. _BDS Pandas:

Pandas
======

Pandas allows access to BDS' HDFS system through :ref: `FSSpec`. This section demonstrates some common operations.

Connect
-------

.. include:: _template/fsspec_connect_with_vault.rst 


File Handle
-----------

You can use the ``fsspec`` `.open() <https://filesystem-spec.readthedocs.io/en/latest/_modules/fsspec/core.html#open>`_ method to open a data file. It returns a file handle. That file handle, ``f``, can be passed to any Pandas' methods that support file handles. In this example, a file on a BDS' HDFS cluster is read into a Pandas dataframe.

.. code-block:: python3

    with fs.open("/data/biketrips/201901-tripdata.csv", "r") as f:
        df = pd.read_csv(f)


URL
---

Pandas supports ``fsspec`` so you can preform file operations by specifying a protocol string. The ``WebHDFS`` protocol is used to access files on BDS' HDFS system. The protocol string has this format:

.. code-block:: python3 

    webhdfs://host:port/path/to/data

The ``host`` and ``port`` parameters can be passed in the protocol string as follows:

.. code-block:: python3

    df = pd.read_csv(f"webhdfs://{hdfs_config['host']}:{hdfs_config['port']}/data/biketrips/201901-tripdata.csv", 
                     storage_options={'kerberos': 'True'})    

You can also pass the ``host`` and ``port`` parameters in the dictionary used by the ``storage_options`` parameter. The sample code for ``hdfs_config`` defines the host and port with the keyes ``host`` and ``port`` respectively.

.. code-block:: python3

    hdfs_config = {
        "protocol": "webhdfs",
        "host": cred["hdfs_host"],
        "port": cred["hdfs_port"],
        "kerberos": "True"
    }

In this case, Pandas uses the following syntax to read a file on BDS' HDFS cluster:

.. code-block:: python3

    df = pd.read_csv(f"webhdfs:///data/biketrips/201901-tripdata.csv", 
                     storage_options=hdfs_config)   


.. _BDS PyArrow:

PyArrow
=======

`PyArrow <https://arrow.apache.org/docs/python/index.html>`_ is a Python interface to `Apache Arrow <https://arrow.apache.org/>`_. Apache Arrow is an in-memory columnar analytical tool that is designed to process data at scale. PyArrow supports the ``fspec.filesystem()`` through the use of the ``filesystem`` parameter in many of its data operation methods.


Connect
-------

Make a connection to BDS' HDFS using ``fsspec``:

.. include:: _template/fsspec_connect_with_vault.rst   


Filesystem
----------

The following sample code shows several different PyArrow methods for working with BDS' HDFS using the ``filesystem`` parameter:

.. code-block:: python3

    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    
    ds = ds.dataset("/path/on/BDS/HDFS/data.csv", format="csv", filesystem=fs)
    pq.write_table(ds.to_table(), '/path/on/BDS/HDFS/data.parquet', filesystem=fs)

    import pandas as pd
    import numpy as np

    idx = pd.date_range('2022-01-01 12:00:00.000', '2022-03-01 12:00:00.000', freq='T')

    df = pd.DataFrame({
            'numeric_col': np.random.rand(len(idx)),
            'string_col': pd._testing.rands_array(8,len(idx))},
            index = idx
        )
    df["dt"] = df.index
    df["dt"] = df["dt"].dt.date

    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(table, root_path="/path/on/BDS/HDFS", partition_cols=["dt"], 
                        flavor="spark", filesystem=fs)

