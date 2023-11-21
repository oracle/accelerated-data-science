from abc import ABC
from typing import Any, Dict

from typing import List

from ads.common.serializer import Serializable, DataClassSerializable


class MarketplaceListingDetails(DataClassSerializable, ABC):
    def __init__(self, listing_id: str, compartment_id: str, version: str):
        self.listing_id = listing_id
        self.compartment_id = compartment_id
        self.version = version
        pass


class HelmMarketplaceListingDetails(MarketplaceListingDetails):
    def __init__(
        self,
        listing_id: str,
        compartment_id: str,
        marketplace_version: str,
        helm_app_name: str,
        namespace: str,
        ocir_repo: str,
        helm_chart_tag: str,
        container_tag_pattern: List[str],
        helm_values: dict,
    ):
        super().__init__(listing_id, compartment_id, marketplace_version)
        self.helm_app_name = helm_app_name
        self.namespace = namespace
        self.ocir_repo = ocir_repo
        self.compartment_id = compartment_id
        self.helm_values = helm_values
        self.helm_chart_tag = helm_chart_tag
        self.container_tag_pattern = container_tag_pattern
