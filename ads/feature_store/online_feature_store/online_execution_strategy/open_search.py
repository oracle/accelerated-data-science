#!/usr/bin/env python
# -*- coding: utf-8; -*-
# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from typing import OrderedDict, Any, Tuple

from pyspark.sql.functions import concat

from ads.feature_store.online_feature_store.online_execution_strategy.online_engine_config.open_search_client_config import (
    OpenSearchClientConfig,
)
from ads.feature_store.online_feature_store.online_feature_store_strategy import (
    OnlineFeatureStoreStrategy,
)


class OnlineOpenSearchEngine(OnlineFeatureStoreStrategy):
    """Online strategy for interacting with OpenSearch for feature serving and embedding retrieval."""

    def __init__(self, online_engine_config: OpenSearchClientConfig):
        self.online_engine_config = online_engine_config

    def write(
        self,
        feature_group,
        feature_group_job,
        dataframe,
        http_auth: Tuple[str, str] = None,
    ):
        """
        Write DataFrame to OpenSearch.

        Parameters:
        - feature_group: Feature group metadata.
        - feature_group_job: Feature group job metadata.
        - dataframe: Spark DataFrame to be written.
        - http_auth: Tuple containing username and password for authentication.

        Raises:
        - ValueError: If http_auth is not provided for authentication.
        """
        index_name = f"{feature_group.entity_id}_{feature_group.name}".lower()
        # Accessing username and password from the tuple
        username = http_auth[0]
        password = http_auth[1]

        primary_keys = [
            key_item["name"] for key_item in feature_group.primary_keys["items"]
        ]
        composite_key = None

        # Elasticsearch configuration

        es_write_config = {
            "opensearch.nodes": self.online_engine_config.host,  # Replace with your Elasticsearch server address
            "opensearch.port": "9200",  # Replace with your Elasticsearch server port
            "opensearch.resource": index_name,  # Replace with your index (without type, as types are deprecated)
            "opensearch.write.operation": "upsert",  # Use "index" for writing data to Elasticsearch
            "opensearch.net.http.auth.user": username,  # Elasticsearch user
            "opensearch.net.http.auth.pass": password,  # Elasticsearch password
            "opensearch.nodes.wan.only": "true",
            "opensearch.net.ssl": "true",
        }

        if len(primary_keys) == 1:
            composite_key = primary_keys[0]
        elif len(primary_keys) > 1:
            composite_key = "composite_key"
            dataframe = dataframe.withColumn(
                composite_key, concat(*[dataframe[col] for col in primary_keys])
            )

        if composite_key is not None:
            es_write_config["opensearch.mapping.id"] = composite_key

        # Write DataFrame to Elasticsearch
        dataframe.write.format("org.opensearch.spark.sql").options(
            **es_write_config
        ).mode(feature_group_job.ingestion_mode).save()

    def read(
        self,
        feature_group,
        keys: OrderedDict[str, Any],
        http_auth: Tuple[str, str] = None,
    ):
        """
        Read data from OpenSearch based on primary key.

        Parameters:
        - feature_group: Feature group metadata.
        - keys: OrderedDict containing primary key information.
        - http_auth: Tuple containing username and password for authentication.

        Returns:
        - dict: Retrieved data from OpenSearch.
        """
        ordered_keys = []
        # TODO:Need to revisit

        for primary_key in feature_group.primary_keys["items"]:
            primary_key_name = primary_key["name"]

            if primary_key_name in keys:
                ordered_keys.append(keys[primary_key_name])
            else:
                raise KeyError(
                    f"'FeatureGroup' object has no primary key called {primary_key_name}."
                )

        ordered_concatenated_key = "".join(ordered_keys)
        response = self.online_engine_config.get_client(http_auth=http_auth).get(
            index=f"{feature_group.entity_id}_{feature_group.name}".lower(),
            id=ordered_concatenated_key,
        )
        return response

    def get_nearest_neighbours(
        self,
        feature_group,
        embedding_field,
        k_neighbors,
        query_embedding_vector,
        max_candidate_pool,
        http_auth: Tuple[str, str] = None,
    ):
        """
        Get nearest neighbors from OpenSearch based on embedding vector.

        Parameters:
        - feature_group: Feature group metadata.
        - embedding_field: Field containing embedding vectors.
        - k_neighbors: Number of neighbors to retrieve.
        - query_embedding_vector: Embedding vector for the query.
        - max_candidate_pool: Maximum number of candidates to consider.
        - http_auth: Tuple containing username and password for authentication.

        Returns:
        - dict: Result of the nearest neighbors search.
        """

        res = self.online_engine_config.get_client(http_auth=http_auth).search(
            index=f"{feature_group.entity_id}_{feature_group.name}".lower(),
            body={
                "size": k_neighbors,
                "query": {
                    "knn": {
                        embedding_field: {
                            "vector": query_embedding_vector,
                            "k": k_neighbors,
                            "candidate_pool_size": max_candidate_pool,
                        }
                    }
                },
            },
        )
        return res
