from ads.feature_store.common.feature_store_singleton import FeatureStoreSingleton
from ads.feature_store.online_feature_store.online_execution_strategy.elastic_search import (
    OnlineElasticSearchEngine,
)
from ads.feature_store.online_feature_store.online_execution_strategy.online_engine_config.elastic_search_client_config import (
    ElasticSearchClientConfig,
)
from ads.feature_store.online_feature_store.online_execution_strategy.online_engine_config.redis_client_config import (
    RedisClientConfig,
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
        feature_store_singleton = FeatureStoreSingleton(feature_store_id)
        online_engine_config = feature_store_singleton.get_online_config()

        if isinstance(online_engine_config, RedisClientConfig):
            return OnlineRedisEngine(online_engine_config)
        elif isinstance(online_engine_config, ElasticSearchClientConfig):
            return OnlineElasticSearchEngine(online_engine_config)
