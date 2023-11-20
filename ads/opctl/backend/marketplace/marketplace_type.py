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
        version: str,
        helm_app_name: str,
        namespace: str,
        ocir_repo: str,
        helm_chart_name: str,
        container_name_pattern: List[str],
        helm_values: dict,
        dcoker_k8_secret_name: str,
    ):
        super().__init__(listing_id, compartment_id, version)
        self.helm_app_name = helm_app_name
        self.namespace = namespace
        self.ocir_repo = ocir_repo
        self.compartment_id = compartment_id
        self.helm_values = helm_values
        self.helm_chart_name = (helm_chart_name,)
        self.container_name_pattern = container_name_pattern
