#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Union

from ads.common.decorator.runtime_dependency import OptionalDependency
from ads.feature_store.common.utils.base64_encoder_decoder import Base64EncoderDecoder
from ads.feature_store.transformation import Transformation, TransformationMode
import pandas as pd

try:
    from pyspark.sql import SparkSession, DataFrame
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `pyspark` module was not found. Please run `pip install "
        f"{OptionalDependency.SPARK}`."
    )
except Exception as e:
    raise


class TransformationUtils:
    @staticmethod
    def apply_transformation(
        spark: SparkSession,
        dataframe: Union[DataFrame, pd.DataFrame],
        transformation: Transformation,
        transformation_kwargs: str,
    ):
        """
        Perform data transformation using either SQL or Pandas, depending on the specified transformation mode.

        Args:
            spark: A SparkSession object.
            transformation (Transformation): A transformation object containing details of transformation to be performed.
            dataframe (DataFrame): The input dataframe to be transformed.
            transformation_kwargs(str): The transformation parameters as json string.

        Returns:
            DataFrame: The resulting transformed data.
        """

        # Fetch the transformation function
        transformation_function = Base64EncoderDecoder.decode(
            transformation.source_code_function
        )

        # Execute the function under namespace
        execution_namespace = {}
        exec(transformation_function, execution_namespace)
        transformation_function_caller = execution_namespace.get(transformation.name)
        transformed_data = None

        transformation_kwargs_dict = json.loads(transformation_kwargs)

        if transformation.transformation_mode == TransformationMode.SQL.value:
            # Register the temporary table
            temporary_table_view = "df_view"
            dataframe.createOrReplaceTempView(temporary_table_view)

            transformed_data = spark.sql(
                transformation_function_caller(
                    temporary_table_view, **transformation_kwargs_dict
                )
            )
        elif transformation.transformation_mode in [
            TransformationMode.PANDAS.value,
            TransformationMode.SPARK.value,
        ]:
            transformed_data = transformation_function_caller(
                dataframe, **transformation_kwargs_dict
            )

        return transformed_data
