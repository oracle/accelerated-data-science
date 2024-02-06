==================
Advanced Use Cases
==================

**Documentation: Anomaly Detection Science and Model Parameterization**

The Science of Anomaly Detection
--------------------------------

Anomaly Detection comes in many forms. We will go through some of these and give guidance as to whether this Operator is going to be helpful for each use case.

* Constructive v Destructive v Pre-Processing: This Operator focuses on the Constructive and Pre-Processing use cases. Destructive can work, but more specific parameters may be required.
* Supervised v Semi-Supervised v Unsupervised: All 3 of these approaches are supported by AutoMLX. AutoTS supports only Unsupervised at this time.
* Time Series. This Operator is focused on just time-series data.


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


**Specify Model Type**

Sometimes users will know which models they want to use. When users know this in advance, they can specify using the ``model_kwargs`` dictionary. In the following example, we will instruct the model to *only* use the ``IsolationForestOD`` model.

.. code-block:: yaml

  kind: operator
  type: anomaly
  version: v1
  spec:
    model: automlx
    model_kwargs:
      model_list:
        - IsolationForestOD
      search_space:
        IsolationForestOD:
          n_estimators:
            range': [10, 50]
            type': 'discrete'


AutoTS offers the same extensibility:

.. code-block:: yaml

  kind: operator
  type: anomaly
  version: v1
  spec:
    model: autots
    model_kwargs:
      method: IQR
