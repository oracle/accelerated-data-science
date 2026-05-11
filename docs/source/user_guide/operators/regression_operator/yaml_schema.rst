===========
YAML Schema
===========

This page walks through the regression operator YAML based on the current implementation in ``ads/opctl/operator/lowcode/regression/schema.yaml`` and the corresponding runtime code.

Complete Example
----------------

.. code-block:: yaml

    kind: operator
    type: regression
    version: v1
    spec:
      training_data:
        url: train.csv
      test_data:
        url: test.csv
      output_directory:
        url: results
      target_column: target
      column_types:
        event_date: date
        customer_id: categorical
      model: random_forest
      model_kwargs:
        tuning_n_trials: 10
        n_estimators: 300
      preprocessing:
        enabled: true
        steps:
          missing_value_imputation: true
          categorical_encoding: true
      metric: rmse
      training_predictions_filename: training_predictions.csv
      test_predictions_filename: test_predictions.csv
      training_metrics_filename: training_metrics.csv
      test_metrics_filename: test_metrics.csv
      global_explanation_filename: global_explanations.csv
      report_filename: report.html
      report_title: Regression Report
      generate_report: true
      generate_explanations: false

Top-Level Fields
----------------

* ``kind``: Must be ``operator``.
* ``type``: Must be ``regression``.
* ``version``: The current schema version is ``v1``.
* ``spec``: Contains the operator-specific configuration.

Data Inputs
-----------

``training_data``
~~~~~~~~~~~~~~~~~

Required. This is the dataset used for fitting the model.

Supported schema fields include:

* ``url``
* ``data``
* ``sql``
* ``table_name``
* ``connect_args``
* ``format``
* ``columns``
* ``options``
* ``limit``
* ``vault_secret_id``

For CLI-first workflows, ``url`` is the normal choice:

.. code-block:: yaml

    training_data:
      url: /path/to/train.csv

Or from Object Storage:

.. code-block:: yaml

    training_data:
      url: oci://bucket@namespace/regression/train.csv

Or from SQL:

.. code-block:: yaml

    training_data:
      sql: |
        SELECT feature_1, feature_2, target
        FROM DEMO.REGRESSION_TRAIN
      connect_args:
        wallet_dir: /home/datascience/oci_wallet

``test_data``
~~~~~~~~~~~~~

Optional. Use this when you want held-out evaluation.

Important:

* The operator always validates that ``test_data`` contains the same feature columns as ``training_data``.
* ``test_metrics.csv`` and ``test_predictions.csv`` are written only when ``test_data`` includes the target column.

``output_directory``
~~~~~~~~~~~~~~~~~~~~

Optional. Defaults to ``results``.

The operator writes artifacts here. Local paths and ``oci://`` paths are both supported.

Target and Feature Typing
-------------------------

``target_column``
~~~~~~~~~~~~~~~~~

Required. This is the continuous value to predict.

``column_types``
~~~~~~~~~~~~~~~~

Optional. Use this to override automatic type inference.

Supported values are:

* ``numerical``
* ``categorical``
* ``date``

Example:

.. code-block:: yaml

    column_types:
      sales_date: date
      zip_code: categorical
      revenue: numerical

If ``column_types`` is not provided, the operator infers feature types from the training data.

Preprocessing
-------------

The current preprocessing implementation supports:

* numeric coercion for numeric-like strings
* median imputation for numeric columns
* mode imputation for categorical columns
* one-hot encoding for categorical columns
* date expansion into ``year``, ``month``, ``day``, ``dayofweek``, and ``dayofyear``

Configuration:

.. code-block:: yaml

    preprocessing:
      enabled: true
      steps:
        missing_value_imputation: true
        categorical_encoding: true

Important cautions
~~~~~~~~~~~~~~~~~~

* If you disable ``categorical_encoding`` while string categorical features are still present, the processed matrix can no longer be converted to numeric form and training can fail.
* If you disable ``preprocessing.enabled``, do so only when your remaining features are already in a model-ready numeric form.

Model Selection
---------------

``model``
~~~~~~~~~

Supported values:

* ``auto``
* ``linear_regression``
* ``random_forest``
* ``knn``
* ``xgboost``

``metric``
~~~~~~~~~~

Supported values:

* ``rmse``
* ``mae``
* ``mse``
* ``r2``
* ``mape``

This metric controls:

* explicit-model tuning
* ``auto`` model selection

The metrics output files still include all five metrics regardless of which one you choose as the primary optimization metric.

``model_kwargs``
~~~~~~~~~~~~~~~~

This dictionary is passed to the explicit model implementation and also supports ``tuning_n_trials``.

Example:

.. code-block:: yaml

    model: knn
    model_kwargs:
      tuning_n_trials: 5
      n_neighbors: 11
      weights: distance

Current behavior:

* Explicit models use Optuna tuning by default with ``20`` trials.
* Setting ``model_kwargs.tuning_n_trials: 0`` disables tuning and uses the current default estimator parameters plus any explicit overrides you provide.
* ``auto`` currently compares candidate models using cross-validation and then retrains the selected model. It does not use user-supplied explicit-model ``model_kwargs`` during candidate comparison.

Output Files
------------

The output filenames can be customized with:

* ``training_predictions_filename``
* ``test_predictions_filename``
* ``training_metrics_filename``
* ``test_metrics_filename``
* ``global_explanation_filename``
* ``report_filename``

The report title can be customized with:

* ``report_title``

Report and Explainability Flags
-------------------------------

``generate_report``
~~~~~~~~~~~~~~~~~~~

Defaults to ``true``. When enabled, the operator writes ``report.html``.

``generate_explanations``
~~~~~~~~~~~~~~~~~~~~~~~~~

Defaults to ``false``.

Current implementation details:

* ``global_explanations.csv`` is generated only when ``generate_explanations`` is ``true``.
* When ``generate_explanations`` is ``true``, the operator first tries model-derived importance for models that expose it.
* If model-derived importance is unavailable, the operator attempts a SHAP-based fallback.
* This SHAP fallback is most relevant to ``knn``.
* If explainability is requested but cannot be produced, the run continues and the report explains that explainability was unavailable for that run.

Deployment Configuration
------------------------

The operator also supports:

.. code-block:: yaml

    save_and_deploy_to_md:
      model_catalog_display_name: regression-model
      project_id: ocid1.datascienceproject.oc1..example
      compartment_id: ocid1.compartment.oc1..example
      model_deployment:
        display_name: regression-md
        initial_shape: VM.Standard.E4.Flex
        description: Regression model deployment
        log_group: ocid1.loggroup.oc1..example
        log_id: ocid1.log.oc1..example
        auto_scaling:
          minimum_instance: 1
          maximum_instance: 2
          scale_in_threshold: 10
          scale_out_threshold: 80
          scaling_metric: CPU_UTILIZATION
          cool_down_in_seconds: 600

When this block is present, the run also writes deployment metadata artifacts. See :doc:`./productionize`.
