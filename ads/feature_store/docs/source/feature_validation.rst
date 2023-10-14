.. _Feature Validation:

Feature Validation
******************

Feature validation is the process of checking the quality and accuracy of the features used in a machine learning model. This is important because features that are not accurate or reliable can lead to poor model performance.
Feature store allows you to define expectation on the data which is being materialized into feature group and dataset. This is achieved using open source library Great Expectations.

.. note::
  `Great Expectations <https://docs.greatexpectations.io/docs/0.15.50/>`_ is a Python-based open-source library for validating, documenting, and profiling your data. It helps you to maintain data quality and improve communication about data between teams. Software developers have long known that automated testing is essential for managing complex codebases. Great Expectations empowers you to define and enforce your data expectations when handling and processing data, allowing for swift detection of data anomalies. In essence, Expectations serve as the equivalent of unit tests for your data, enabling you to rapidly identify and address data-related problems. Beyond this, Great Expectations offers the added benefit of generating comprehensive data documentation and quality reports based on these Expectations.

.. image:: figures/data_validation.png

Expectations
============
An Expectation is a verifiable assertion about your data. You can define expectation as below:

.. code-block:: python3

    from great_expectations.core.expectation_configuration import ExpectationConfiguration

    # Create an Expectation
    expect_config = ExpectationConfiguration(
        # Name of expectation type being added
        expectation_type="expect_table_columns_to_match_ordered_list",
        # These are the arguments of the expectation
        # The keys allowed in the dictionary are Parameters and
        # Keyword Arguments of this Expectation Type
        kwargs={
            "column_list": [
                "column1",
                "column2",
                "column3",
                "column4",
            ]
        },
        # This is how you can optionally add a comment about this expectation.
        meta={
            "notes": {
                "format": "markdown",
                "content": "details about this expectation. **Markdown** `Supported`",
            }
        },
    )

Expectations Suite
===================

Expectation Suite is a collection of verifiable assertions i.e. expectations about your data. You can define expectation suite as below:

.. code-block:: python3

    # Create an Expectation Suite
    expectation_suite = ExpectationSuite(
        expectation_suite_name=<expectation_suite_name>
    )
    expectation_suite.add_expectation(expect_config)
