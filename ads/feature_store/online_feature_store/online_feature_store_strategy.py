from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any


class OnlineFeatureStoreStrategy(ABC):
    """
    An abstract base class that defines a strategy for ingesting and deleting feature definitions and datasets.
    """

    @abstractmethod
    def write(self, feature_group, feature_group_job, dataframe):
        pass

    @abstractmethod
    def read(self, feature_group, primary_key_vector):
        pass

    # TODO :Yogesh to verify
    @abstractmethod
    def read(
        self,
        feature_group,
        keys: OrderedDict[str, Any],
    ):
        pass
