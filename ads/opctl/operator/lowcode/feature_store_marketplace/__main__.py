import sys

from ads.opctl.backend.marketplace.marketplace_operator_runner import (
    MarketplaceOperatorRunner,
)

from ads.opctl.backend.marketplace.marketplace_type import (
    MarketplaceListingDetails,
    HelmMarketplaceListingDetails,
)


class FeatureStoreOperatorRunner(MarketplaceOperatorRunner):
    def get_listing_details(self, operator_config: str) -> MarketplaceListingDetails:
        return HelmMarketplaceListingDetails(
            listing_id="sada",
            helm_values="{}",
            cluster_id="sds",
            namespace="asdas",
            ocir_repo="sad",
        )


if __name__ == "__main__":
    print(sys.argv)
    FeatureStoreOperatorRunner().run(sys.argv)
