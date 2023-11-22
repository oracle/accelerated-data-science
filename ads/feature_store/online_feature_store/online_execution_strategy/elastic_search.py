from ads.feature_store.online_feature_store.online_feature_store_strategy import OnlineFeatureStoreStrategy


class OnlineElasticSearchEngine(OnlineFeatureStoreStrategy):
    def write(self, feature_group, data_frame):
        # Elasticsearch configuration
        es_conf = {
            "es.nodes": "localhost",  # Replace with your Elasticsearch server address
            "es.port": "9200",  # Replace with your Elasticsearch server port
            "es.resource": "your_index/your_type",  # Replace with your index and type
            "es.write.operation": "index",  # Use "index" for writing data to Elasticsearch
            "es.net.http.auth.user": "elastic",  # Replace with your Elasticsearch username
            "es.net.http.auth.pass": "43ef9*ixWnJbsiclO*lU",  # Replace with your Elasticsearch password
            "es.net.ssl": "true",  # Enable SSL/TLS
            "es.net.ssl.cert.allow.self.signed": "true",  # Allow self-signed certificates
            "es.net.ssl.ca": "/Users/yogeshkumawat/Downloads/elasticsearch-8.11.1/config/certs/http_ca.crt"
            # Specify CA certificate
        }

        # Write DataFrame to Elasticsearch
        data_frame.write.format("org.elasticsearch.spark.sql").options(**es_conf).save()