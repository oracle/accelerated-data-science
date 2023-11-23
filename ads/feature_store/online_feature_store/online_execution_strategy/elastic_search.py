from pyspark.sql.functions import concat

from ads.feature_store.online_feature_store.online_feature_store_strategy import (
    OnlineFeatureStoreStrategy,
)


class OnlineElasticSearchEngine(OnlineFeatureStoreStrategy):
    def write(self, feature_group, feature_group_job, dataframe):
        index_name = f"{feature_group.entity_id}_{feature_group.name}"


        primary_keys = feature_group.primary_keys["items"]
        composite_key = None
        if len(primary_keys) == 1:
            composite_key = primary_keys[0]
        else:
            composite_key = "composite_key"
            dataframe = dataframe.withColumn(composite_key, concat(*[dataframe[col] for col in primary_keys]))


        # Elasticsearch configuration
        elastic_search_config = {
            "es.nodes": "localhost",  # Replace with your Elasticsearch server address
            "es.port": "9200",  # Replace with your Elasticsearch server port
            "es.resource": index_name,  # Replace with your index (without type, as types are deprecated)
            "es.write.operation": "index",  # Use "index" for writing data to Elasticsearch
            "es.mapping.id": "id",  # Specify the mapping ID TODO: Need to revisit
            "es.net.http.auth.user": "elastic",  # Elasticsearch user
            "es.net.http.auth.pass": "43ef9*ixWnJbsiclO*lU",  # Elasticsearch password
            "es.nodes.wan.only": "true",
            "es.nodes.discovery": "false",
            "es.nodes.client.only": "false",
            "es.write.operation": "upsert",
        }

        # Write DataFrame to Elasticsearch
        dataframe.write.format("org.elasticsearch.spark.sql").options(
            **elastic_search_config
        ).mode(feature_group_job.ingestion_mode).save()
