from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any


class OnlineFeatureStoreStrategy(ABC):
    """
    An abstract base class that defines a strategy for ingesting and deleting feature definitions and datasets.
    """

    @abstractmethod
    def write(self, feature_group_or_dataset, feature_group_or_dataset_job, dataframe):
        pass

    @abstractmethod
    def read(self, feature_group_or_dataset, primary_key_vector):
        pass

    def get_embedding_vector(
        self,
        feature_group_or_dataset,
        embedding_field,
        k_neighbors,
        query_embedding_vector,
        max_candidate_pool,
    ):
        pass
