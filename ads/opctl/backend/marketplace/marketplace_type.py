from abc import ABC

from ads.common.serializer import Serializable


class MarketplaceListingDetails(Serializable, ABC):
    def __init__(self, listing_id: str):
        self.listing_id = listing_id
        pass


class HelmMarketplaceListingDetails(ABC):
    def __init__(
        self, listing_id: str, cluster_id: str, namespace: str, ocir_repo: str
    ):
        self.listing_id = listing_id
        self.cluster_id = cluster_id
        self.namespace = namespace
        self.ocir_repo = ocir_repo
