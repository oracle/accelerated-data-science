#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging

from ads.common.decorator.runtime_dependency import OptionalDependency
from ads.feature_store.common.spark_session_singleton import SparkSessionSingleton
from ads.feature_store.statistics_config import StatisticsConfig

try:
    import pydeequ
    from pydeequ.profiles import *
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `pydeequ` module was not found. Please run `pip install "
        f"{OptionalDependency.PYDEEQU}`."
    )

logger = logging.getLogger(__name__)


class StatisticsService:
    """StatisticsService is used to compute the statistics using pydeequ column profiler"""

    @staticmethod
    def compute_statistics(
        spark: SparkSession, statistics_config: StatisticsConfig, input_df: DataFrame
    ):
        feature_statistics = None
        if (
            bool(input_df.head(1))
            and statistics_config
            and statistics_config.get("isEnabled")
        ):
            logger.info("Calculating metrics")
            relevant_columns = statistics_config.get("columns")
            stat_runner = ColumnProfilerRunner(spark).onData(input_df)
            if relevant_columns:
                stat_runner.restrictToColumns(relevant_columns)
            result = stat_runner.run()
            column_profiles = {}
            for col, profile in result.profiles.items():
                column_profile_dict = None
                if isinstance(profile, StandardColumnProfile):
                    column_profile_dict = {
                        "completeness": profile.completeness,
                        "approximateNumDistinctValues": profile.approximateNumDistinctValues,
                        "dataType": profile.dataType,
                    }
                if isinstance(profile, NumericColumnProfile):
                    column_profile_dict = {
                        "completeness": profile.completeness,
                        "approximateNumDistinctValues": profile.approximateNumDistinctValues,
                        "dataType": profile.dataType,
                        "sum": profile.sum,
                        "min": profile.minimum,
                        "max": profile.maximum,
                        "mean": profile.mean,
                        "stddev": profile.stdDev,
                    }
                column_profiles[col] = column_profile_dict
            feature_statistics = json.dumps(column_profiles)
        return feature_statistics
