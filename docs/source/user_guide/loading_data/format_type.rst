``DatasetFactory`` Formats
**************************

You can load data with different formats into ``DatasetFactory``, see **Loading Data** in :ref:`Loading Data <loading-data-10>`.  Following are some examples.

ARFF
====

You can load ARFF file into ``DatasetFactory``. The file format is recognized from the file name. You can load the file from internet:

.. code-block:: python3

    ds = DatasetFactory.open('https://*example.com/path/to/some_data.arff*')

Array
=====

You can convert an array into a Pandas DataFrame and then open it with ``DatasetFactory``:

.. code-block:: python3

  generated_data_arr = [["ID", "Name", "GPA"], [1, "Bob", 3.7], [2, "Sam", 4.3], [3, "Erin", 2.6]]
  generated_df1 = pd.DataFrame(generated_data_arr[1:], columns=generated_data_arr[0])
  generated_ds1 = DatasetFactory.open(generated_df1)

Delimited Files
===============

CSV and TSV are the most common delimited files. However, files can have other forms of delimitation. To read them with the ``DatasetFactory.open()`` method, the delimiter parameter must be given with the delimiting value. ``DatasetFactory.open()`` considers all delimited files as CSV so the ``format=csv`` or ``format=csv`` parameter must also be specified even though the delimiter is not a comma or tab. ``DatasetFactory.open()`` attempts to determine the column names from the first line of the file. Alternatively, the ``column_names`` option can be used to specify them.

In this example, a file is created that is delimited with a vertical bar (|), and then read in with the ``DatasetFactory.open()`` method.

.. code-block:: python3

  # Create a delimited file with a '|' as a separator
  file = tempfile.NamedTemporaryFile()
  for i in range(5):
      for j in range(7):
          term = '|' if j != 6 else '\n'
          file.write(bytes('{}.{}'.format(i, j) + term, 'utf-8'))
  file.flush()

  # Print the raw file
  file.seek(0)
  for line in file:
      print(line.decode("utf-8"))

  # Read in the delimited file and specify the column names.
  ds = DatasetFactory.open(file.name, delimiter='|', format='csv', column_names=['a','b','c','d','e','f'])
  file.close()
  ds.head()

CSV
---

You can load a ``csv`` file into Dataset Factory using ``open()``:

.. code-block:: python3

  ds = DatasetFactory.open("data/multiclass_fk_10k.csv")


.. note::
   If your dataset does not include a header, then ``DatasetFactory`` assumes that each feature is named according to the corresponding column from your first data-point. This feature naming may be undesirable and could lead to subtle bugs appearing. Many CSVs use spaces for readability, which can lead to trouble when trying to set your target variable within ``DatasetFactory.open()``.

   The work around for this is to pass ``header=None`` to ``DatasetFactory``:

   .. code-block:: python3

      ds = DatasetFactory.open("sample_data.csv", header=None)

   All of your columns are given integer names beginning with 1.

TSV
---

You can open a ``tsv`` or a file with any arbitrary separation key with ``DatasetFactory``, using ``open()``. This is an example of a ``tsv`` file being generated and opening it with ``DatasetFactory``:

.. code-block:: python3

  f = open("tmp_random_ds99.tsv","w+")
  f.write('1 \t 2 \t 3 \t 4 \t 5 \t 6 \n 1.1 \t 2.1 \t 3.1 \t 4.1 \t 5.1 \t 6.1')
  f.close()

  ds = DatasetFactory.open("tmp_random_ds99.tsv", column_names=['a','b','c','d','e','f'])


Dictionary
==========

You can convert a dictionary into a Pandas DataFrame and then open it with ``DatasetFactory``:

.. code-block:: python3

  generated_data_dict = {"ID": [1.1, 2.0, 3.0],
                       "Name": ["Bob", "Sam", "Erin"],
                       "GPA": [3.7, 4.3, 2.6]}
  generated_df2 = pd.DataFrame(generated_data_dict)
  generated_ds2 = DatasetFactory.open(generated_df2)

Excel
=====

Data scientists often have to work with Excel files as a data source. If the file extension is ``.xlsx``, then ``DatasetFactory.open()`` automatically processes it as an Excel file. If not, the ``format=xlsx`` can be used. By default, the first sheet in the file is read in. This behavior can be modified with the ``sheetname`` parameter. It accepts the sheet number (it is zero-indexed) or a string with the name of the sheet. ``DatasetFactory.open()`` reads in all columns that have values. This behavior can be modified with the ``usecols`` parameter. It accepts a list of column numbers to be read in, such as usecols=[1, 3, 5] or it can accept a range as a string, ``usecols=A:C``.

.. code-block:: python3

  # Create the Excel file to read in. Put the data on a sheet called 'wine'
  file = tempfile.NamedTemporaryFile()
  writer = pd.ExcelWriter(file.name, engine='xlsxwriter')
  DatasetBrowser.sklearn().open('wine').to_pandas().to_excel(writer, sheet_name='wine')
  writer.save()

  # Read in the Excel file and clean up
  ds = DatasetFactory.open(file.name, format='xlsx', sheetname='wine', usecols="A:C")
  file.close()
  ds.head()

HDF
====

You can load an HDF file into ``DatasetFactory``. This example builds an HDF file, and then opens it with ``DatasetFactory``:

.. code-block:: python3

    [ds_loc] = ds.to_hdf("tmp_random_ds99.h5", key='df')
    ds_copy = DatasetFactory.open(ds_loc, key='df')

JSON
====

JSON files are supported by ``DatasetFactory.open()`` as long as the data can be restructured into a rectangular form. There are two supported formats of JSON that are called orientations. The orientation is given by ``orient=index`` or ``orient=records``.

For the index orientation, there is a single JSON object. The format is:

.. code-block:: python3

  {
      <index>: <value>,
      <index>: <value>
  }

For example:

.. code-block:: python3

  {
      "946684800000": {"id": 982, "name": "Yvonne", "x": -0.3289461521, "y": -0.4301831275},
      "946684801000": {"id": 1031, "name": "Charlie", "x": 0.9002882524, "y": -0.2144513329}
  }

For the records format, there is a collection of JSON objects. No index value is given and there is no comma between records. The format is:

.. code-block:: python3

  {<key>: <value>, <key>: <value>}
  {<key>: <value>, <key>: <value>}

For example:

.. code-block:: python3

  {"id": 982, "name": "Yvonne", "x": -0.3289461521, "y": -0.4301831275}
  {"id": 1031, "name": "Charlie", "x": 0.9002882524, "y": -0.2144513329}

In this example, a JSON file is created then read back in with ``DatasetFactory.open()``. If the file extension ends in ``.json``, then the method loads it as a JSON file. If this is not the case, then set ``format=json``.

.. code-block:: python3

  # Create the JSON file that is to be read
  [file] = DatasetBrowser.sklearn().open('wine').to_json(path.join(tempfile.mkdtemp(), "wine.json"),
                                                         orient='records')

  # Read in the JSON file
  ds = DatasetFactory.open(file, format='json', orient='records')
  ds.head()

Pandas
======

You can pass the ``pandas.DataFrame`` object directly into the ADS ``DatasetFactory.open`` method:

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


You can also use a ``Pandas.DataFrame`` in the same way. `More Pandas information <https://pandas.pydata.org/>`__.

Parquet
========

You can read Parquet files in ADS. This example builds a Parquet folder, and then opens it with ``DatasetFactory``:

.. code-block:: python3

    ds.to_parquet("tmp_random_ds99")


.. code-block:: python3

    ds_copy = DatasetFactory.open("tmp_random_ds99", format='parquet')


.. _loading-data-specify-dtype:

Specify Data Types
******************

When you open a dataset, ADS detects data types in the dataset. The ADS semantic dtypes assigned to features in dataset, can be:

* Categorical
* Continuous
* Datetime
* Ordinal

ADS semantic dtypes are based on ADS low-level dtypes. They match with the Pandas dtypes 'object', 'int64', 'float64', 'datetime64', 'category', and so on. When you use an ``open()`` statement for a dataset, ADS detects both its semantic and low-level data types. This example specifies the low-level data type, and then ADS detects its semantic type:

.. code-block:: python3

    import pandas as pd
    from ads.dataset.factory import DatasetFactory

    df = pd.DataFrame({
            'numbers': [5.0, 6.0, 8.0, 5.0],
            'years': [2007, 2008, 2008, 2009],
            'target': [1, 2, 3, 3]
    })

    ds = DatasetFactory.open(
            df,
            target = 'numbers',
            types = {'numbers': 'int64'}
    )

You can inspect low level and semantic ADS dtypes with the ``feature_types`` property:

.. code-block:: python3

    # print out detailed information on each column
    ds.feature_types

    # print out ADS "semantic" dtype of a column
    print(ds.feature_types['numbers']['type'])

    # print out ADS "low-level" dtype of a column
    print(ds.feature_types['numbers']['low_level_type'])

.. parsed-literal::

    ordinal
    int64

You can also get the summary information on a dataset, including its feature details in a notebook output cell with ``show_in_notebook``:

.. code-block:: python3

    ds.show_in_notebook()

Use `numpy.dtype <https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.html#numpy.dtype>`_ or `Pandas dtypes <https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#dtypes>`_ in ``types`` parameter to specify your data type. When you update a type, ADS changes both the semantic and the low-level types.

You can either specify a semantic or a low-level data type for ``types``. This example shows how to load a dataset with various types of data:

.. code-block:: python3

    ds = DatasetFactory.open(
            df,
            target = 'years',
            types = {'years': 'datetime'}
    )
    print(ds.feature_types['years']['type'])
    print(ds.feature_types['years']['low_level_type'])

.. parsed-literal::

    datetime
    datetime64[ns]

.. code-block:: python3

    ds = DatasetFactory.open(
            df,
            target = 'target',
            types = {'target': 'categorical'}
    )
    print(ds.feature_types['target']['type'])
    print(ds.feature_types['target']['low_level_type'])

.. parsed-literal::

    categorical
    category


You can find more examples about how to change column data types in :ref:`data-transformations-change-dtype`.

