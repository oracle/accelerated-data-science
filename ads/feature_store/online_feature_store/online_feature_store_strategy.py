from abc import ABC, abstractmethod


class OnlineFeatureStoreStrategy(ABC):
    """
    An abstract base class that defines a strategy for ingesting and deleting feature definitions and datasets.
    """

    @abstractmethod
    def write(
        self, feature_group, dataframe
    ):
        pass

    @abstractmethod
    def read(
        self,
        feature_group,
        dataframe,
    ):
        pass
   