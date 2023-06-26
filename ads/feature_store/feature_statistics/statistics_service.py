#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import json
import logging

from ads.feature_store.statistics_config import StatisticsConfig
from ads.feature_store.common.utils.feature_schema_mapper import *

try:
    from pyspark.sql import DataFrame
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `pyspark` module was not found. Please run `pip install "
        f"{OptionalDependency.SPARK}`."
    )

try:
    from mlm_insights.builder.builder_component import EngineDetail
    from mlm_insights.builder.insights_builder import InsightsBuilder
    from mlm_insights.constants.types import FeatureType
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `mlm_insights` module was not found. Please run `pip install "
        f"{OptionalDependency.MLM_INSIGHTS}`."
    )

logger = logging.getLogger(__name__)
CONST_FEATURE_METRICS = "feature_metrics"


class StatisticsService:
    """StatisticsService is used to compute the statistics using pydeequ column profiler"""

    @staticmethod
    def compute_stats_with_mlm(
        statistics_config: StatisticsConfig, input_df: DataFrame
    ):
        feature_metrics = None
        if (
            bool(input_df.head(1))
            and statistics_config
            and statistics_config.is_enabled
        ):
            feature_schema = {}
            if input_df.schema:
                StatisticsService.__get_mlm_supported_schema(
                    feature_schema, input_df, statistics_config
                )
                feature_metrics = StatisticsService.__get_feature_metric(
                    feature_schema, input_df
                )
            else:
                raise ValueError("Dataframe schema is missing")
        return feature_metrics

    @staticmethod
    def __get_feature_metric(feature_schema: dict, data_frame: DataFrame):
        feature_metrics = None
        runner = (
            InsightsBuilder()
            .with_input_schema(input_schema=feature_schema)
            .with_data_frame(data_frame=data_frame)
            .with_engine(engine=EngineDetail(engine_name="spark"))
            .build()
        )
        result = runner.run()
        if result and result.profile:
            profile = result.profile
            feature_metrics = json.dumps(profile.to_json()[CONST_FEATURE_METRICS])
        else:
            logger.warning(
                f"stats computation failed with MLM for schema {feature_schema}"
            )
        return feature_metrics

    @staticmethod
    def __get_mlm_supported_schema(
        feature_schema: dict, input_df: DataFrame, statistics_config: StatisticsConfig
    ):
        relevant_columns = statistics_config.columns
        for field in input_df.schema.fields:
            data_type = map_spark_type_to_stats_data_type(field.dataType)
            if not data_type:
                logger.warning(
                    f"Unable to map spark data type fields to MLM fields, "
                    f"Actual data type {field.dataType}"
                )
            elif relevant_columns:
                if field.name in relevant_columns:
                    feature_schema[field.name] = FeatureType(
                        data_type, map_spark_type_to_stats_variable_type(field.dataType)
                    )
            else:
                feature_schema[field.name] = FeatureType(
                    data_type, map_spark_type_to_stats_variable_type(field.dataType)
                )
