from pyspark.sql.functions import col, concat_ws

from ads.feature_store.online_feature_store.online_feature_store_strategy import OnlineFeatureStoreStrategy


class OnlineRedisEngine(OnlineFeatureStoreStrategy):
    def write(
            self, feature_group, dataframe
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