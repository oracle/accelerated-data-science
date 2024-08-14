=================
Data Integration
=================

Supported Data Sources
----------------------

The Operator can read data from the following sources:
- Oracle RDBMS
- OCI Object Storage
- OCI Data Lake
- HTTPS
- S3
- Azure Blob Storage
- Google Cloud Storage
- Local file systems

Additionally, the operator supports any data source supported by `fsspec <https://filesystem-spec.readthedocs.io/en/latest/_modules/fsspec/registry.html>`_.

Examples
--------

Reading from OCI Object Storage
===============================

Below is an example of reading data from OCI Object Storage using the operator:

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

Reading from Oracle Database
============================

Below is an example of reading data from an Oracle Database:

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

The forecasting operator simplifies powerful data preprocessing. By default, it includes several preprocessing steps to ensure dataset compliance with each framework. However, users can disable one or more of these steps if needed, though doing so may cause the model to fail. Proceed with caution.

Default preprocessing steps:
- Missing value imputation
- Outlier treatment

To disable ``outlier_treatment``, modify the YAML file as shown below:

.. code-block:: yaml

    kind: operator
    type: forecast
    version: v1
    spec:
        datetime_column:
            name: ds
        historical_data:
            url: https://raw.githubusercontent.com/facebook/prophet/main/examples/example_yosemite_temps.csv
        horizon: 3
        target_column: y
        preprocessing: 
            enabled: true
            steps:
                missing_value_imputation: True
                outlier_treatment: False


Real-Time Trigger
-----------------

The Operator can be run locally or on an OCI Data Science Job. The resultant model can be saved and deployed for future use if needed. For questions regarding this integration, please reach out to the OCI Data Science team.
