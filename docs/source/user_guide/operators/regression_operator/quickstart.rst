===========
Quick Start
===========

Install
-------

Install the dependencies listed in :doc:`./install` first.

Initialize
----------

Generate starter configs with the ADS CLI:

.. code-block:: bash

   ads operator init -t regression --overwrite --output ~/regression/

The generated files always include:

* ``regression.yaml``
* ``regression_operator_local_python_backend.yaml``
* ``regression_operator_local_container_backend.yaml``

If your ADS CLI defaults are configured for OCI Data Science Jobs, ``init`` also generates:

* ``regression_job_container_backend.yaml``
* ``regression_job_python_backend.yaml``

Prepare the YAML
----------------

Open ``~/regression/regression.yaml`` and fill in the training data, optional labeled test data, optional unlabeled prediction data, target column, and output directory.

Example:

.. code-block:: yaml

    kind: operator
    type: regression
    version: v1
    spec:
      training_data:
        url: /path/to/train.csv
      test_data:
        url: /path/to/test.csv
      prediction_data:
        url: /path/to/rows_to_score.csv
      prediction_output:
        passthrough_columns: [record_id, series_id, period]
      output_directory:
        url: /path/to/results
      target_column: target
      model: linear_regression
      model_kwargs:
        tuning_n_trials: 0
      generate_report: true
      generate_explanations: false

Why ``tuning_n_trials: 0`` in the example?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current explicit-model implementations use Optuna-backed tuning by default. Setting ``tuning_n_trials: 0`` makes the first run faster and easier to validate.

Verify
------

Validate the configuration before running:

.. code-block:: bash

    ads operator verify -f ~/regression/regression.yaml

Run Locally
-----------

Run the operator in the local python backend:

.. code-block:: bash

    ads operator run -f ~/regression/regression.yaml -b local

Artifacts
---------

For a run with both training and test data, you should expect:

* ``training_predictions.csv``
* ``test_predictions.csv``
* ``predictions.csv`` when ``prediction_data`` is configured
* ``training_metrics.csv``
* ``test_metrics.csv``
* ``report.html`` when ``generate_report: true``
* ``model.pkl``

``predictions.csv`` contains the configured passthrough columns and ``prediction`` for every input row, preserving input order. If ``passthrough_columns`` is omitted, all ``prediction_data`` columns are included by default. A unique row key is not required. It is produced without registration or deployment unless ``save_and_deploy_to_md`` is configured.

If you also set ``generate_explanations: true``, the run can additionally produce ``global_explanations.csv`` and ``local_explanations.csv``. For example, the checked-in regression test asset produces prediction and metric outputs like:

.. code-block:: text

    input_value,predicted_value,residual
    13.0,12.94857982370225,0.051420176297749975
    14.6,14.525857002938292,0.07414299706170802

And training metrics like:

.. code-block:: text

    metrics,target
    sMAPE,0.55
    MAPE,0.010921881463703724
    RMSE,0.26522709702026454
    r2,0.9853933943119197
    Explained Variance,0.9853933943119193
    MAE,0.1846327130264453
    MSE,0.07034541299379685

Open the HTML report after the run:

.. code-block:: bash

    open /path/to/results/report.html
