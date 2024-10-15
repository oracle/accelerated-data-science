==================
Advanced Use Cases
==================

The Science of Anomaly Detection
--------------------------------

Anomaly Detection comes in many forms. We will go through some of these and give guidance as to whether this Operator is going to be helpful for each use case.

* Constructive v Destructive v Pre-Processing: This Operator focuses on the Constructive and Pre-Processing use cases. Destructive can work, but more specific parameters may be required.
* The operator currently supports only unsupervised learning and works with both time-series and non-time-series data.


Data Parameterization
---------------------

**Read Data from the Database**

.. code-block:: yaml

    kind: operator
    type: anomaly
    version: v1
    spec:
        input_data:
            connect_args:
                user: XXX
                password: YYY
                dsn: "localhost/orclpdb"
            sql: 'SELECT Store_ID, Sales, Date FROM live_data'
        datetime_column:
            name: ds
        target_column: y


**Read Part of a Dataset**


.. code-block:: yaml

    kind: operator
    type: anomaly
    version: v1
    spec:
        input_data:
            url: oci://bucket@namespace/data
            format: hdf
            limit: 1000  # Only the first 1000 rows
            columns: ["y", "ds"]  # Ignore other columns
        datetime_column:
            name: ds
        target_column: y
