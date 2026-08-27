===
FAQ
===

Why did I get training outputs but no explanation CSV files?
-------------------------------------------------------------

The operator generates ``global_explanations.csv`` and ``local_explanations.csv`` when ``generate_explanations: true``.

Current behavior:

* If ``generate_explanations`` is ``false``, no explanation files are written.
* Global importance uses a model-derived value when the selected model exposes one and otherwise uses SHAP.
* Local explanations use SHAP for every supported model and explain a deterministic sample of up to 200 training rows. ``base_value`` plus the signed transformed-feature contributions reconstructs each ``prediction``, subject to rounding.
* Keep ``generate_explanations: true`` and make sure ``shap`` is installed in the operator runtime.

Why did I not get ``test_metrics.csv``?
---------------------------------------

``test_metrics.csv`` is written only when:

* ``test_data`` is provided
* the test dataset includes the same feature columns as training
* the test dataset also includes ``target_column``

Why does a simple explicit-model run take longer than expected?
---------------------------------------------------------------

Explicit models use Optuna-based tuning by default. If you want a faster validation run, set:

.. code-block:: yaml

    model_kwargs:
      tuning_n_trials: 0

Why does ``auto`` not use my explicit ``model_kwargs``?
-------------------------------------------------------

In the current implementation, ``auto`` compares the candidate models using its own candidate-selection path and then retrains the selected model. User-supplied explicit-model ``model_kwargs`` are not used during the candidate comparison stage.

When should I override ``column_types``?
----------------------------------------

Override ``column_types`` when automatic inference is likely to be misleading, for example:

* identifier-like numeric columns such as ``customer_id`` or ``zip_code``
* date columns stored as strings
* numeric values stored as text with inconsistent formatting

Why did training fail after I disabled preprocessing?
-----------------------------------------------------

The current preprocessing pipeline always converts the final feature matrix to numeric form before fitting. If you disable preprocessing or categorical encoding while raw string categorical columns remain, the matrix may no longer be numeric and model training can fail.
