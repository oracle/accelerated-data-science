from abc import ABC, abstractmethod
from ads.opctl.backend.marketplace.marketplace_type import MarketplaceListingDetails


class MarketplaceInterface(ABC):
    @abstractmethod
    def get_listing_details(
        self,
        operator_config: str,
    ) -> MarketplaceListingDetails:
        pass
