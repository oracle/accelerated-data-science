from abc import ABC, abstractmethod

from typing import Dict

from ads.opctl.backend.marketplace.models.marketplace_type import (
    MarketplaceListingDetails,
)


# TODO: Handle generic listings properly
class MarketplaceInterface(ABC):
    @abstractmethod
    def get_listing_details(self, operator_config: str) -> MarketplaceListingDetails:
        pass

    @abstractmethod
    def get_oci_meta(self, container_map: Dict[str, str], operator_config: str) -> dict:
        pass
