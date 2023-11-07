from abc import ABC

from ads.common.serializer import Serializable, DataClassSerializable


class MarketplaceListingDetails(DataClassSerializable, ABC):
    def __init__(self, listing_id: str):
        self.listing_id = listing_id
        pass


class HelmMarketplaceListingDetails(MarketplaceListingDetails):
    def __init__(
            self,
            listing_id: str,
            name: str,
            chart: str,
            version: str,
            cluster_id: str,
            namespace: str,
            ocir_repo: str,
            helm_values,
    ):
        super().__init__(listing_id)
        self.cluster_id = cluster_id
        self.name = name
        self.chart = chart
        self.version = version
        self.namespace = namespace
        self.ocir_repo = ocir_repo
        self.helm_values = helm_values
