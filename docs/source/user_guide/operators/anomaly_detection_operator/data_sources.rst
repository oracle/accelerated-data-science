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
    type: anomaly
    version: v1
    spec:
        datetime_column:
            name: ds
        historical_data:
            url: oci://<bucket_name>@<namespace_name>/example_yosemite_temps.csv
        target_column: y

Example: Reading from Oracle Database
=====================================

.. code-block:: yaml

    kind: operator
    type: anomaly
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
        target_column: y


Data Preprocessing
------------------

Operators are enabled to be powerful through simplicity. The forecasting operator provides several pre-processing steps by default to ensure the dataset is compliant with each framework. Occasionally a user may know better and want to disable one or more of these steps. This may cause the modeling to fail, proceed with caution.
The steps are:
- missing_value_imputation
- outlier_treatment

To disable ``outlier_treatment``, amend the yaml file as shown below:

.. code-block:: yaml

    kind: operator
    type: anomaly
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


Real Time Trigger
-----------------

The Operator runs locally or on an OCI Data Science Job. The resultant model can be saved and deployed for future use if need be. Please reach out to the OCI Data Science team with any questions surrounding this integration.
