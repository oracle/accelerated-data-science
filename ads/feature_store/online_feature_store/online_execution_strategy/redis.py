#!/usr/bin/env python
# -*- coding: utf-8; -*-
# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from typing import OrderedDict, Any

from pyspark.sql.functions import col, concat_ws

from ads.feature_store.online_feature_store.online_execution_strategy.online_engine_config.redis_client_config import (
    RedisClientConfig,
)
from ads.feature_store.online_feature_store.online_feature_store_strategy import (
    OnlineFeatureStoreStrategy,
)


class OnlineRedisEngine(OnlineFeatureStoreStrategy):
    def __init__(self, online_engine_config: RedisClientConfig):
        self.online_engine_config = online_engine_config
        self.redis_client = online_engine_config.get_client()

    def write(
        self,
        feature_group,
        feature_group_job,
        dataframe,
        http_auth: tuple[str, str] = None,
    ):
        if len(feature_group.primary_keys["items"]) == 1:
            key = feature_group.primary_keys["items"][0]["name"]
            df_with_key = dataframe.withColumn("key", col(key))
        else:
            primary_keys = []
            for key in feature_group.primary_keys["items"]:
                primary_keys.append(key["name"])
            df_with_key = dataframe.withColumn("key", concat_ws(":", *primary_keys))
        df_with_key.write.format("org.apache.spark.sql.redis").option(
            "table", feature_group.id
        ).option("key.column", "key").save()

    def read(
        self,
        feature_group,
        keys: OrderedDict[str, Any],
        http_auth: tuple[str, str] = None,
    ):
        ordered_keys = []
        # TODO:Need to move this out and generalize
        for primary_key in feature_group.primary_keys["items"]:
            primary_key_name = primary_key["name"]
            if primary_key_name in keys:
                ordered_keys.append(keys[primary_key_name])
            else:
                raise KeyError(
                    f"'FeatureGroup' object has no primary key called {primary_key_name}."
                )
        ordered_concatenated_key = ":".join(ordered_keys)
        response = self.redis_client.hgetall(
            f"{feature_group.id}:{ordered_concatenated_key}"
        )
        return response

    def get_embedding_vector(
        self,
        feature_group,
        embedding_field,
        k_neighbors,
        query_embedding_vector,
        max_candidate_pool,
        http_auth: tuple[str, str] = None,
    ):
        raise NotImplementedError(
            "The method get_embedding_vector is not supported for Redis."
        )
