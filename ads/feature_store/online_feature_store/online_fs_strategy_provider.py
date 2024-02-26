#!/usr/bin/env python
# -*- coding: utf-8; -*-
# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from ads.feature_store.common.feature_store_singleton import FeatureStoreSingleton
from ads.feature_store.online_feature_store.online_execution_strategy.open_search import (
    OnlineOpenSearchEngine,
)
from ads.feature_store.online_feature_store.online_execution_strategy.online_engine_config.open_search_client_config import (
    OpenSearchClientConfig,
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
        elif isinstance(online_engine_config, OpenSearchClientConfig):
            return OnlineOpenSearchEngine(online_engine_config)
