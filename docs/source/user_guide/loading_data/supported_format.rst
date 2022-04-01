Supported Formats
-----------------

You can load datasets into ADS, either locally or from network file systems.

You can open datasets with ``DatasetFactory``, ``DatasetBrowser`` or ``pandas``. ``DatasetFactory`` allows datasets to be loaded into ADS.

``DatasetBrowser`` supports opening the datasets from web sites and libraries, such as scikit-learn directly into ADS.

When you open a dataset in ``DatasetFactory``, you can get the summary statistics, correlations, and visualizations of the dataset.

ADS Supports:

+-------------------+-----------------------------------------------+
| **Data Sources**  | Oracle Cloud Infrastructure Object Storage    |
|                   +-----------------------------------------------+
|                   | Oracle Database with cx_Oracle                |
|                   +-----------------------------------------------+
|                   | Autonomous Databases: ADW and ATP             |
|                   +-----------------------------------------------+
|                   | Hadoop Distributed File System                |
|                   +-----------------------------------------------+
|                   | Amazon S3                                     |
|                   +-----------------------------------------------+
|                   | Google Cloud Service                          |
|                   +-----------------------------------------------+
|                   | Microsoft Azure                               |
|                   +-----------------------------------------------+
|                   | Blob                                          |
|                   +-----------------------------------------------+
|                   | MongoDB                                       |
|                   +-----------------------------------------------+
|                   | NoSQL DB instances                            |
|                   +-----------------------------------------------+
|                   | Elastic Search instances                      |
|                   +-----------------------------------------------+
|                   | HTTP and HTTPs Sources                        |
|                   +-----------------------------------------------+
|                   | Your local files                              |
+-------------------+-----------------------------------------------+
| **Data Formats**  | Pandas.DataFrame, Dask.DataFrame              |
|                   +-----------------------------------------------+
|                   | Array, Dictionary                             |
|                   +-----------------------------------------------+
|                   | Comma Separated Values (CSV)                  |
|                   +-----------------------------------------------+
|                   | Tab Separated Values (TSV)                    |
|                   +-----------------------------------------------+
|                   | Parquet                                       |
|                   +-----------------------------------------------+
|                   | Javascript Object Notation (JSON)             |
|                   +-----------------------------------------------+
|                   | XML                                           |
|                   +-----------------------------------------------+
|                   | xls, xlsx (Excel)                             |
|                   +-----------------------------------------------+
|                   | LIBSVM                                        |
|                   +-----------------------------------------------+
|                   | Hierarchical Data Format 5 (HDF5)             |
|                   +-----------------------------------------------+
|                   | Apache server log files                       |
|                   +-----------------------------------------------+
|                   | HTML                                          |
|                   +-----------------------------------------------+
|                   | Avro                                          |
|                   +-----------------------------------------------+
|                   | Attribute-Relation File Format (ARFF)         |
+-------------------+-----------------------------------------------+
| **Data Types**    | Text Types (`str`)                            |
|                   +-----------------------------------------------+
|                   | Numeric Types (`int`, `float`)                |
|                   +-----------------------------------------------+
|                   | Boolean Types (`bool`)                        |
+-------------------+-----------------------------------------------+

ADS *Does Not* Support:

+-------------------+-----------------------------------------------+
| **Data Sources**  | Data that you don't have permissions to.      |
+-------------------+-----------------------------------------------+
| **Data Formats**  | Text Files                                    |
|                   +-----------------------------------------------+
|                   | DOCX                                          |
|                   +-----------------------------------------------+
|                   | PDF                                           |
|                   +-----------------------------------------------+
|                   | Raw Images                                    |
|                   +-----------------------------------------------+
|                   | SAS                                           |
+-------------------+-----------------------------------------------+
| **Data Types**    | Sequence Types (`list`, `tuple`, `range`)     |
|                   +-----------------------------------------------+
|                   | Mapping Types (`dict`)                        |
|                   +-----------------------------------------------+
|                   | Set Types (`set`)                             |
+-------------------+-----------------------------------------------+

For reading text files, DOCX and PDF, see "Text Extraction" section.
