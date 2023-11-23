from ads.feature_store.online_feature_store.online_execution_strategy.elastic_search import (
    OnlineElasticSearchEngine,
)
from ads.feature_store.online_feature_store.online_execution_strategy.redis import (
    OnlineRedisEngine,
)
from ads.feature_store.online_feature_store.online_feature_store_strategy import (
    OnlineFeatureStoreStrategy,
)


class OnlineFSStrategyProvider:
    @classmethod
    def provide_online_execution_strategy(
        cls, feature_store_id: str
    ) -> OnlineFeatureStoreStrategy:
        from ads.feature_store.feature_store import FeatureStore

        feature_store = FeatureStore.from_id(feature_store_id)

        if feature_store.online_config["redisId"]:
            return OnlineRedisEngine()
        elif feature_store.online_config["elasticSearchId"]:
            return OnlineElasticSearchEngine()
