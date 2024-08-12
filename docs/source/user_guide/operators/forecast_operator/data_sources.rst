================
Data Integration
================

Data Sources
------------

The Operator can read data from:
- Oracle RDBMS
- OCI Object Storage
- OCI Data Lake
- HTTPS
- S3
- Azure Blob Storage
- Google Cloud Storage
- local

The operator supports any data source supported by `fsspec <https://filesystem-spec.readthedocs.io/en/latest/_modules/fsspec/registry.html>`_.

Example: Reading from Object Storage
=====================================

.. code-block:: yaml

    kind: operator
    type: forecast
    version: v1
    spec:
        datetime_column:
            name: ds
        historical_data:
            url: oci://<bucket_name>@<namespace_name>/example_yosemite_temps.csv
        horizon: 3
        target_column: y

Example: Reading from Oracle Database
=====================================

.. code-block:: yaml

    kind: operator
    type: forecast
    version: v1
    spec:
        historical_data:
            connect_args:
                user: XXX
                password: YYY
                dsn: "localhost/orclpdb"
            sql: 'SELECT Store_ID, Sales, Date FROM live_data'
        datetime_column:
            name: ds
        horizon: 1
        target_column: y


Data Preprocessing
------------------

Operators are enabled to be powerful through the simplicity. Pre-processing is fundamental to this mission. By pre-processing 
