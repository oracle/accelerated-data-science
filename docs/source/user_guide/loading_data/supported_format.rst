Supported Formats
*****************

You can load datasets into ADS, either locally or from network file systems.

You can open datasets with ``DatasetFactory``, ``DatasetBrowser`` or ``pandas``. ``DatasetFactory`` allows datasets to be loaded into ADS.

``DatasetBrowser`` supports opening the datasets from web sites and libraries, such as scikit-learn directly into ADS.

When you open a dataset in ``DatasetFactory``, you can get the summary statistics, correlations, and visualizations of the dataset.

ADS Supports:

+-------------------+-----------------------------------------------+
| **Data Sources**  | Amazon S3                                     |
|                   +-----------------------------------------------+
|                   | Autonomous Databases: ADW and ATP             |
|                   +-----------------------------------------------+
|                   | Blob                                          |
|                   +-----------------------------------------------+
|                   | Elastic Search instances                      |
|                   +-----------------------------------------------+
|                   | Google Cloud Service                          |
|                   +-----------------------------------------------+
|                   | HTTP and HTTPs Sources                        |
|                   +-----------------------------------------------+
|                   | Hadoop Distributed File System                |
|                   +-----------------------------------------------+
|                   | Local files                                   |
|                   +-----------------------------------------------+
|                   | Microsoft Azure                               |
|                   +-----------------------------------------------+
|                   | MongoDB                                       |
|                   +-----------------------------------------------+
|                   | NoSQL DB instances                            |
|                   +-----------------------------------------------+
|                   | Oracle Cloud Infrastructure Object Storage    |
|                   +-----------------------------------------------+
|                   | Oracle Database with cx_Oracle                |
+-------------------+-----------------------------------------------+
| **Data Formats**  | Apache server log files                       |
|                   +-----------------------------------------------+
|                   | Array, Dictionary                             |
|                   +-----------------------------------------------+
|                   | Attribute-Relation File Format (ARFF)         |
|                   +-----------------------------------------------+
|                   | Avro                                          |
|                   +-----------------------------------------------+
|                   | Comma Separated Values (CSV)                  |
|                   +-----------------------------------------------+
|                   | HTML                                          |
|                   +-----------------------------------------------+
|                   | Hierarchical Data Format 5 (HDF5)             |
|                   +-----------------------------------------------+
|                   | Javascript Object Notation (JSON)             |
|                   +-----------------------------------------------+
|                   | LIBSVM                                        |
|                   +-----------------------------------------------+
|                   | Pandas.DataFrame, Dask.DataFrame              |
|                   +-----------------------------------------------+
|                   | Parquet                                       |
|                   +-----------------------------------------------+
|                   | PDF                                           |
|                   +-----------------------------------------------+
|                   | Tab Separated Values (TSV)                    |
|                   +-----------------------------------------------+
|                   | xls, xlsx (Excel)                             |
|                   +-----------------------------------------------+
|                   | XML                                           |
+-------------------+-----------------------------------------------+
| **Data Types**    | Boolean Types (`bool`)                        |
|                   +-----------------------------------------------+
|                   | Numeric Types (`int`, `float`)                |
|                   +-----------------------------------------------+
|                   | Text Types (`str`)                            |
+-------------------+-----------------------------------------------+

ADS *Does Not* Support:

+-------------------+-----------------------------------------------+
| **Data Formats**  | DOCX                                          |
|                   +-----------------------------------------------+
|                   | Raw Images                                    |
|                   +-----------------------------------------------+
|                   | SAS                                           |
|                   +-----------------------------------------------+
|                   | Text Files                                    |
+-------------------+-----------------------------------------------+
| **Data Types**    | Mapping Types (`dict`)                        |
|                   +-----------------------------------------------+
|                   | Set Types (`set`)                             |
|                   +-----------------------------------------------+
|                   | Sequence Types (`list`, `tuple`, `range`)     |
+-------------------+-----------------------------------------------+

For reading text files, DOCX and PDF, see "Text Extraction" section.

