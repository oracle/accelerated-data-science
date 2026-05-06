===================
Regression Operator
===================

The Regression Operator is a low-code operator for supervised tabular regression. It trains a model from a training dataset, optionally evaluates on held-out test data, and writes a consistent set of artifacts such as predictions, metrics, an HTML report, and a serialized model bundle.

Overview
--------

**Required inputs**

The current implementation requires:

* ``training_data``
* ``target_column``

All columns in ``training_data`` except ``target_column`` are treated as features.

**Optional inputs**

The operator also supports:

* ``test_data`` for held-out evaluation
* ``output_directory`` for artifact location
* ``column_types`` to override automatic type inference
* ``model_kwargs`` to control explicit model runs
* ``save_and_deploy_to_md`` to save the trained model to OCI Model Catalog and create a Model Deployment

**Supported models**

The supported ``model`` values are:

* ``auto``
* ``linear_regression``
* ``random_forest``
* ``knn``
* ``xgboost``

``auto`` performs cross-validation across the explicit model families and selects the best one for the configured ``metric``. Explicit models use Optuna-based tuning by default.

**Preprocessing**

By default, the operator:

* infers numeric, categorical, and date columns
* imputes missing numeric values with the median
* imputes missing categorical values with the mode
* one-hot encodes categorical columns
* expands date columns into ``year``, ``month``, ``day``, ``dayofweek``, and ``dayofyear``

**Artifacts**

Depending on the configuration and available data, the operator can write:

* ``training_predictions.csv``
* ``test_predictions.csv``
* ``training_metrics.csv``
* ``test_metrics.csv``
* ``global_explanations.csv``
* ``report.html``
* ``model.pkl``
* ``model_registration_info.json``
* ``deployment_info.json``

``global_explanations.csv`` is written only when ``generate_explanations: true`` and explainability output is successfully produced.

.. toctree::
  :maxdepth: 1

  ./quickstart
  ./install
  ./yaml_schema
  ./advanced_use_cases
  ./productionize
  ./faq
