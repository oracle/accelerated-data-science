#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.feature_store.common.enums import ExecutionEngine
from ads.feature_store.execution_strategy.spark.spark_execution import (
    SparkExecutionEngine,
)
from ads.feature_store.execution_strategy.execution_strategy import Strategy


class OciExecutionStrategyProvider:
    """
    A class that provides an execution strategy based on the specified execution engine.

    Methods:
        provide_execution_strategy: Returns an execution strategy for the specified execution engine.
    """

    @classmethod
    def provide_execution_strategy(
        cls, execution_engine: ExecutionEngine, metastore_id: str = None
    ) -> "Strategy":
        """
        Returns an execution strategy for the specified execution engine.

        Args:
            execution_engine (ExecutionEngine): The execution engine to use.
            metastore_id (str): The metastore_id which will be used to create the spark.

        Returns:
            Strategy: An execution strategy for the specified execution engine.

        Raises:
            ValueError: If the specified execution engine is not supported.
        """

        if execution_engine == ExecutionEngine.SPARK:
            return SparkExecutionEngine(metastore_id)
        elif execution_engine == ExecutionEngine.PANDAS:
            return SparkExecutionEngine(metastore_id)
        else:
            raise ValueError(
                "Unsupported execution engine: {}".format(execution_engine)
            )
