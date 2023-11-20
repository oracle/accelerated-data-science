import json
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
        operator_config_json = json.loads(operator_config)
        operator_config_spec = operator_config_json["spec"]
        listing_id="ocid1.mktpublisting.oc1.iad.amaaaaaaclen5bqas6lyd5xler6fewsri5uascfhybhrtwjnh3fwyzbzaora"
        return HelmMarketplaceListingDetails(
            listing_id=listing_id,
            helm_chart_name="",
            container_name_pattern=[""],
            version="v0.10.2",
            helm_values=operator_config_spec["helmValues"],
            namespace=operator_config_spec["clusterDetails"]["namespace"],
            dcoker_k8_secret_name="",
            ocir_repo=operator_config_spec["ocirRepo"],
            compartment_id="ocid1.compartment.oc1..aaaaaaaa3nvibvakxapvbd46rr3nclxb2kmop7moppqnfdnkpdcafziumygq",
            helm_app_name="fs-dp",
        )


if __name__ == "__main__":
    print(sys.argv)
    FeatureStoreOperatorRunner().run(sys.argv)
