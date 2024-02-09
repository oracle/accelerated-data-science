#!/usr/bin/env python
# -*- coding: utf-8; -*-
# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import ads
from ads.common.decorator.runtime_dependency import OptionalDependency
import os
from ads.common.oci_client import OCIClientFactory
from ads.feature_store.online_feature_store.online_execution_strategy.online_engine_config.open_search_client_config import (
    OpenSearchClientConfig,
)
from ads.feature_store.online_feature_store.online_execution_strategy.online_engine_config.redis_client_config import (
    RedisClientConfig,
)

try:
    from delta import configure_spark_with_delta_pip
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `pyspark` module was not found. Please run `pip install "
        f"{OptionalDependency.SPARK}`."
    )
except Exception as e:
    raise
try:
    from pyspark.sql import SparkSession
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `pyspark` module was not found. Please run `pip install "
        f"{OptionalDependency.SPARK}`."
    )
except Exception as e:
    raise


def get_env_bool(env_var: str, default: bool = False) -> bool:
    """
    :param env_var: Environment variable name
    :param default: Default environment variable value
    :return: Value of the boolean env variable
    """
    env_val = os.getenv(env_var)
    if env_val is None:
        env_val = default
    else:
        env_val = env_val.lower()
        if env_val == "true":
            env_val = True
        elif env_val == "false":
            env_val = False
        else:
            raise ValueError(
                "For environment variable: {0} only string values T/true or F/false are allowed but: \
                {1} was provided.".format(
                    env_var, env_val
                )
            )
    return env_val


def developer_enabled():
    return get_env_bool("DEVELOPER_MODE", False)


def redis_path():
    return os.getenv("REDIS_PATH")


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        feature_store_id = kwargs.get("feature_store_id", None)
        key = (cls, feature_store_id)
        if key not in cls._instances:
            cls._instances[key] = super().__call__(*args, **kwargs)
        return cls._instances[key]


class FeatureStoreSingleton(metaclass=SingletonMeta):
    """Class provides the spark session."""

    def __init__(self, feature_store_id: str = None):
        """Virtually private constructor."""

        spark_builder = (
            SparkSession.builder.appName("FeatureStore")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .enableHiveSupport()
        )
        _managed_table_location = None
        fs_online_config = None
        feature_store = None

        if feature_store_id:
            from ads.feature_store.feature_store import FeatureStore

            # Parse the Feature Store and get the required Details
            feature_store = FeatureStore.from_id(feature_store_id)
            offline_config = feature_store.offline_config

            if feature_store.online_config:
                fs_online_config = self.__get_feature_store_online_config(
                    feature_store.online_config
                )
            metastore_id = offline_config["metastoreId"]

            if not developer_enabled() and metastore_id:
                # Configure the Spark session builder to use the specified metastore
                _managed_table_location = self.__configure_metastore(
                    spark_builder, metastore_id
                )

        if developer_enabled():
            self.spark_session = configure_spark_with_delta_pip(
                spark_builder,
                ## org.opensearch.client:opensearch-spark-30_2.12:1.0.1
                extra_packages=[
                    "org.opensearch.client:opensearch-spark-30_2.12:1.0.1",
                    "com.redislabs:spark-redis_2.12:3.0.0",
                ],
            ).getOrCreate()
        else:
            self.spark_session = spark_builder.getOrCreate()

        self.spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        self.spark_session.sparkContext.setLogLevel("OFF")
        self.managed_table_location = _managed_table_location
        self.online_config = fs_online_config
        self.feature_store = feature_store

    def __configure_metastore(self, spark_builder, metastore_id):
        """Configures the Spark session builder with the specified metastore."""
        # Fetch authentication credentials for the OCI data catalog service
        auth = copy.copy(ads.auth.default_signer())
        auth.pop("client_kwargs", None)

        data_catalog_client = OCIClientFactory(**auth).data_catalog
        metastore = data_catalog_client.get_metastore(metastore_id).data
        _managed_table_location = metastore.default_managed_table_location

        spark_builder.config(
            "spark.hadoop.oracle.dcat.metastore.id", metastore_id
        ).config("spark.sql.warehouse.dir", _managed_table_location).config(
            "spark.driver.memory", "16G"
        )

        return _managed_table_location

    def __get_feature_store_online_config(self, online_config):
        auth = copy.copy(ads.auth.default_signer())
        auth.pop("client_kwargs", None)
        client_factory = OCIClientFactory(**auth)

        if online_config.get("openSearchId"):
            auth = copy.copy(ads.auth.default_signer())
            auth.pop("client_kwargs", None)

            opensearch_client = client_factory.opensearch
            get_opensearch_cluster_response = opensearch_client.get_opensearch_cluster(
                opensearch_cluster_id=online_config.get("openSearchId")
            )
            opensearch_endpoint_host = (
                get_opensearch_cluster_response.data.opensearch_fqdn
            )

            return OpenSearchClientConfig(
                host=opensearch_endpoint_host,
                verify_certs=False,
            )

        elif online_config.get("redisId"):
            redis_client = client_factory.redis
            get_redis_cluster_response = redis_client.get_redis_cluster(
                redis_cluster_id=online_config.get("redisId")
            )

            redis_endpoint_host = get_redis_cluster_response.data.primary_fqdn

            return RedisClientConfig(host=redis_endpoint_host, port=6379)

        return None

    def get_spark_session(self):
        """Access method to get the spark session."""
        return self.spark_session

    def get_managed_table_location(self):
        """Returns the managed table location for the spark"""
        return self.managed_table_location

    def get_online_config(self):
        return self.online_config

    def get_feature_store(self):
        return self.feature_store
