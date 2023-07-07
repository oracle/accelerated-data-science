Feature Validation
*************

Save expectation entity
=======================
With a ``FeatureGroup`` or ``Dataset`` instance, we can save the expectation entity using ``save_expectation()``

.. note::

  Great Expectations is a Python-based open-source library for validating, documenting, and profiling your data. It helps you to maintain data quality and improve communication about data between teams. Software developers have long known that automated testing is essential for managing complex codebases.

.. image:: figures/validation.png

The ``.save_expectation()`` method takes the following optional parameter:

- ``expectation: Expectation``. Expectation of great expectation
- ``expectation_type: ExpectationType``. Type of expectation
        - ``ExpectationType.STRICT``: Fail the job if expectation not met
        - ``ExpectationType.LENIENT``: Pass the job even if expectation not met

.. code-block:: python3

  feature_group.save_expectation(expectation_suite, expectation_type="STRICT")
  dataset.save_expectation(expectation_suite, expectation_type="STRICT")

