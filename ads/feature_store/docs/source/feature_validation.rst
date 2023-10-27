.. _Feature Validation:

Feature Validation
******************

Feature validation is the process of checking the quality and accuracy of the features used in a machine learning model. This is important because features that aren't accurate or reliable can lead to poor model performance. Feature store allows you to define expectation on the data that is being materialized into feature groups and datasets. The Great Expectations open source library is used to define expectations.

.. note::
  `Great Expectations <https://docs.greatexpectations.io/docs/0.15.50/>`_  is an open source Python-based library that validates, documents, and profiles data. It automates testing, which is essential for managing complex code bases.

.. image:: figures/data_validation.png

Expectations
============
An expectation is a verifiable assertion about your data.

The following example defines an expectation:

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

An expectation suite is a collection of verifiable assertions. For example, expectations about your data.

The following example defines an expectation suite:

.. code-block:: python3

    # Create an Expectation Suite
    expectation_suite = ExpectationSuite(
        expectation_suite_name=<expectation_suite_name>
    )
    expectation_suite.add_expectation(expect_config)
