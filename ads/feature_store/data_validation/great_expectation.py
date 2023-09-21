#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import uuid

import pandas as pd

from ads.common.decorator.runtime_dependency import OptionalDependency

try:
    from great_expectations.core import ExpectationConfiguration
    from great_expectations.core import ExpectationSuite, IDDict
    from great_expectations.core.batch import BatchDefinition, Batch
    from great_expectations.execution_engine import (
        SparkDFExecutionEngine,
        PandasExecutionEngine,
    )
    from great_expectations.validator.validator import Validator
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `feature-store` module was not found. Please run `pip install "
        f"{OptionalDependency.FEATURE_STORE}`."
    )
except Exception as e:
    raise
from ads.feature_store.common.enums import ExpectationType


class ExpectationService:
    """A utility class for defining and validating data quality expectations on dataframes using Great Expectations.

    This class provides methods to define a set of data quality expectations based on a JSON string, add them to an
    existing expectation suite, and validate a dataframe against the updated expectation suite. It also includes a
    method to apply data quality validations on a dataframe and raise an exception if any expectation fails.

    Methods
    -------
    apply_validations(expectation_details, expectation_suite_name, dataframe):
        Applies a set of data quality validations to a dataframe based on a set of expectation rules parsed from a JSON
        string. If any expectation fails, an exception is raised. This method can be used to enforce strict data quality
        requirements on a dataframe.
    """

    @staticmethod
    def __add_validation_to_expectation_suite(
        expectation_suite: ExpectationSuite, dataframe, expectations_rules
    ):
        # Parse the JSON string into a list of expectations
        # Iterate over the list of expectations and add them to the expectation suite
        for expect in expectations_rules:
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type=expect["ruleType"].lower(),
                    kwargs=expect["arguments"],
                )
            )

        # Define the execution engine based on the dataframe type
        execution_engine = (
            PandasExecutionEngine()
            if isinstance(dataframe, pd.DataFrame)
            else SparkDFExecutionEngine(force_reuse_spark_context=True)
        )
        # Validate the dataframe against the updated expectation suite
        validator = Validator(
            execution_engine=execution_engine,
            expectation_suite=expectation_suite,
            batches=[
                Batch(
                    data=dataframe,
                    batch_definition=BatchDefinition(
                        datasource_name="feature-ingestion-pipeline",
                        data_connector_name="feature-ingestion-pipeline",
                        data_asset_name="feature-ingestion-pipeline",
                        batch_identifiers=IDDict(ge_batch_id=str(uuid.uuid1())),
                    ),
                ),
            ],
        )
        validation_result = validator.validate()
        return validation_result

    @staticmethod
    def __validate_expectation_details(
        expectation_details, feature_group, input_dataframe
    ):
        # Initialize Expectation Suite
        suite = ExpectationSuite(expectation_suite_name=feature_group)
        suite.expectations = []

        expectations_rules = expectation_details["createRuleDetails"]
        expectation_response = ExpectationService.__add_validation_to_expectation_suite(
            suite, input_dataframe, expectations_rules
        )
        return expectation_response

    @staticmethod
    def apply_validations(expectation_details, expectation_suite_name, dataframe):
        """Validate the dataframe against the expectations in expectation_details.

        Parameters
        ----------
        expectation_details : dict
            The details of the expectations.
        expectation_suite_name : str
            The name of the expectation suite.
        dataframe : Union[pd.DataFrame, pyspark.sql.DataFrame]
            The data to validate.

        Returns
        -------
        str
            A string representation of the validation result.
        """
        expectation_response = None
        if (
            expectation_details
            and expectation_details.get("expectationType")
            != ExpectationType.NO_EXPECTATION
        ):
            # Validate the Validations
            expectation_response = ExpectationService.__validate_expectation_details(
                expectation_details, expectation_suite_name, dataframe
            )

        return expectation_response
