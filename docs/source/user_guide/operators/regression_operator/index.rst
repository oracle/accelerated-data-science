===================
Regression Operator
===================

The Regression Operator is a low-code operator for supervised tabular regression. It trains a model from a training dataset, optionally evaluates on labeled held-out test data, scores a separate unlabeled batch dataset, and writes a consistent set of artifacts such as predictions, metrics, an HTML report, and a serialized model bundle. Batch prediction does not require model registration or deployment.

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
* ``prediction_data`` and ``prediction_output`` for unlabeled batch scoring; all prediction input columns pass through by default
* ``output_directory`` for artifact location
* ``column_types`` to override automatic type inference
* ``model_kwargs`` to control explicit model runs
* ``save_and_deploy_to_md`` for Model Catalog registration and Model Deployment creation

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
* ``predictions.csv``
* ``training_metrics.csv``
* ``test_metrics.csv``
* ``global_explanations.csv``
* ``report.html``
* ``model.pkl``
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
