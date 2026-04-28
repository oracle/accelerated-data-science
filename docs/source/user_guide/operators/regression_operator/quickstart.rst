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

Open ``~/regression/regression.yaml`` and fill in the training data, optional test data, target column, and output directory.

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
* ``training_metrics.csv``
* ``test_metrics.csv``
* ``report.html`` when ``generate_report: true``
* ``model.pkl``

If you also set ``generate_explanations: true``, the run can additionally produce ``global_explanations.csv``. For example, the checked-in regression test asset produces prediction and metric outputs like:

.. code-block:: text

    input_value,predicted_value,residual
    13.0,12.94857982370225,0.051420176297749975
    14.6,14.525857002938292,0.07414299706170802

And training metrics like:

.. code-block:: text

    metric,value
    rmse,0.2652270970202646
    mae,0.1846327130264453
    mse,0.07034541299379685
    r2,0.9853933943119193
    mape,1.0921881463703744

Open the HTML report after the run:

.. code-block:: bash

    open /path/to/results/report.html
