from ads.feature_store.online_feature_store.online_feature_store_strategy import (
    OnlineFeatureStoreStrategy,
)


class OnlineElasticSearchEngine(OnlineFeatureStoreStrategy):
    def write(self, feature_group, feature_group_job, dataframe):
        # Elasticsearch configuration
        elastic_search_conf = {
            "es.nodes": "localhost",  # Replace with your Elasticsearch server address
            "es.port": "9200",  # Replace with your Elasticsearch server port
            "es.resource": "index_name",  # Replace with your index (without type, as types are deprecated) #TODO: Need to revisit
            "es.write.operation": "index",  # Use "index" for writing data to Elasticsearch
            "es.mapping.id": "id",  # Specify the mapping ID
            "es.net.http.auth.user": "elastic",  # Elasticsearch user
            "es.net.http.auth.pass": "43ef9*ixWnJbsiclO*lU",  # Elasticsearch password
        }

        # Write DataFrame to Elasticsearch
        dataframe.write.format("org.elasticsearch.spark.sql").options(
            **elastic_search_conf
        ).mode(feature_group_job.ingestion_mode).save()
