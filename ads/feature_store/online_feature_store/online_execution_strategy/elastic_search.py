from typing import OrderedDict, Any, Dict

from pyspark.sql.functions import concat

from ads.feature_store.online_feature_store.online_feature_store_strategy import (
    OnlineFeatureStoreStrategy,
)


class OnlineElasticSearchEngine(OnlineFeatureStoreStrategy):
    def write(self, feature_group, feature_group_job, dataframe):
        index_name = f"{feature_group.entity_id}_{feature_group.name}".lower()
        primary_keys = [
            key_item["name"] for key_item in feature_group.primary_keys["items"]
        ]
        composite_key = None

        # Elasticsearch configuration
        elastic_search_config = {
            "es.nodes": "localhost",  # Replace with your Elasticsearch server address
            "es.port": "9200",  # Replace with your Elasticsearch server port
            "es.resource": index_name,  # Replace with your index (without type, as types are deprecated)
            "es.write.operation": "index",  # Use "index" for writing data to Elasticsearch
            "es.net.http.auth.user": "elastic",  # Elasticsearch user
            "es.net.http.auth.pass": "43ef9*ixWnJbsiclO*lU",  # Elasticsearch password
            "es.nodes.wan.only": "true",
            "es.nodes.discovery": "false",
            "es.nodes.client.only": "false",
            "es.write.operation": "upsert",
        }

        if len(primary_keys) == 1:
            composite_key = primary_keys[0]
        elif len(primary_keys) > 1:
            composite_key = "composite_key"
            dataframe = dataframe.withColumn(
                composite_key, concat(*[dataframe[col] for col in primary_keys])
            )

        if composite_key is not None:
            elastic_search_config["es.mapping.id"] = composite_key

        # Write DataFrame to Elasticsearch
        dataframe.write.format("org.elasticsearch.spark.sql").options(
            **elastic_search_config
        ).mode(feature_group_job.ingestion_mode).save()

    def read(self, feature_group, keys: OrderedDict[str, Any]):
        ordered_keys = []
        # TODO:Need to revisit
        from elasticsearch import Elasticsearch

        elastic_search_client = Elasticsearch(
            hosts="http://localhost:9200",
            basic_auth=("elastic", "43ef9*ixWnJbsiclO*lU"),
            verify_certs=False,
        )

        for primary_key in feature_group.primary_keys["items"]:
            primary_key_name = primary_key["name"]

            if primary_key_name in keys:
                ordered_keys.append(keys[primary_key_name])
            else:
                raise KeyError(
                    f"'FeatureGroup' object has no primary key called {primary_key_name}."
                )

        ordered_concatenated_key = "".join(ordered_keys)
        response = elastic_search_client.get(
            index=f"{feature_group.entity_id}_{feature_group.name}".lower(),
            id=ordered_concatenated_key,
        )
        return response
