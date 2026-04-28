==================
Advanced Use Cases
==================

Mixed Feature Types
-------------------

The regression operator automatically handles mixed tabular inputs.

Example:

.. code-block:: yaml

    kind: operator
    type: regression
    version: v1
    spec:
      training_data:
        url: train.csv
      target_column: target
      model: random_forest
      column_types:
        event_date: date
        customer_id: categorical

With the current implementation:

* numeric-like strings are coerced to numeric values
* categorical columns are one-hot encoded
* date columns are expanded to ``year``, ``month``, ``day``, ``dayofweek``, and ``dayofyear``

This is useful when a CSV contains values such as:

* ``numeric_text`` columns stored as strings
* identifier columns such as ``customer_id``
* date strings such as ``2025-01-01``

Reading from Object Storage or SQL
----------------------------------

Object Storage:

.. code-block:: yaml

    training_data:
      url: oci://bucket@namespace/regression/train.csv
    test_data:
      url: oci://bucket@namespace/regression/test.csv

SQL:

.. code-block:: yaml

    training_data:
      sql: |
        SELECT x1, x2, x3, target
        FROM DEMO.REGRESSION_TRAIN
      connect_args:
        wallet_dir: /home/datascience/oci_wallet
    test_data:
      sql: |
        SELECT x1, x2, x3, target
        FROM DEMO.REGRESSION_TEST
      connect_args:
        wallet_dir: /home/datascience/oci_wallet

Explicit Tuning Control
-----------------------

Explicit models use Optuna-backed tuning by default.

To reduce runtime for development:

.. code-block:: yaml

    model: linear_regression
    model_kwargs:
      tuning_n_trials: 0

To keep tuning enabled but bounded:

.. code-block:: yaml

    model: xgboost
    model_kwargs:
      tuning_n_trials: 5
      n_estimators: 300
      max_depth: 6

Current tuning behavior:

* ``model_kwargs`` act as fixed overrides
* the remaining model-specific parameters can still be explored by Optuna
* the selected ``metric`` controls optimization direction

Understanding ``auto``
----------------------

The ``auto`` model currently compares:

* ``linear_regression``
* ``random_forest``
* ``knn``
* ``xgboost``

It evaluates them with cross-validation on the training data using the configured ``metric`` and then retrains the selected model on the full training set.

Example:

.. code-block:: yaml

    model: auto
    metric: rmse

Important current behavior:

* ``auto`` uses default candidate configurations during candidate comparison
* user-supplied explicit-model ``model_kwargs`` are not used during this selection stage

Explainability by Model Family
------------------------------

``linear_regression``
~~~~~~~~~~~~~~~~~~~~~

Global explanations come from absolute coefficient values.

``random_forest`` and ``xgboost``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Global explanations come from model-derived feature importances.

``knn``
~~~~~~~

KNN does not expose built-in feature importance. If you want explainability output for KNN, enable SHAP-based explanations:

.. code-block:: yaml

    model: knn
    generate_explanations: true

And make sure ``shap`` is installed in the runtime environment.

Held-Out Evaluation
-------------------

When ``test_data`` contains the same feature columns and also includes the target column:

* ``test_predictions.csv`` is written
* ``test_metrics.csv`` is written
* the HTML report includes a held-out evaluation section

Example:

.. code-block:: yaml

    test_data:
      url: test.csv

If the target column is not present in ``test_data``, the operator still validates feature compatibility but does not generate held-out regression metrics.
